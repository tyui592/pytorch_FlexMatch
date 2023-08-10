"""Training Code."""

import time
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from network import get_network
from data import get_dataloaders
from optim import get_optimizer
from evaluate import Metric, evaluate_step
from collections import defaultdict, Counter
from utils import AverageMeter, mapping_func
from ema import EMA


def train_step(model,
               ema,
               criterion,
               optimizer,
               scheduler,
               num_classes,
               threshold,
               unsupervised_weight,
               amp_flag,
               scaler,
               X,
               U,
               N,
               learning_status,
               mapping,
               device):
    """Train one epoch."""
    global global_step

    logs = defaultdict(AverageMeter)
    metric = Metric()

    cls_thresholds = torch.zeros(num_classes, device=device)

    model.train()
    for sample_x, sample_u in zip(X, U):
        with torch.autocast(device_type='cuda',
                            dtype=torch.float16,
                            enabled=amp_flag):
            # (weak, strong) augmented data
            (xw, _), y, _ = sample_x
            (uw, us), _, u_i = sample_u

            inputs = torch.cat([xw, uw, us], dim=0)
            outputs = model(inputs.to(device))

            xw_pred, uw_pred, us_pred = torch.split(outputs,
                                                    [xw.shape[0],
                                                     uw.shape[0],
                                                     us.shape[0]])

            # supervised loss
            ls = criterion(xw_pred, y.to(device)).mean()
            total_loss = ls

            # compute a learning status
            counter = Counter(learning_status)

            # normalize the status
            num_unused = counter[-1]
            if num_unused != N:
                max_counter = max([counter[c] for c in range(num_classes)])
                if max_counter < num_unused:
                    # normalize with eq.11
                    sum_counter = sum([counter[c] for c in range(num_classes)])
                    denominator = max(max_counter, N - sum_counter)
                else:
                    denominator = max_counter
                # threshold per class
                for c in range(num_classes):
                    beta = counter[c] / denominator
                    cls_thresholds[c] = mapping(beta) * threshold

            # update the pseudo label
            with torch.no_grad():
                uw_prob = softmax(uw_pred, dim=1)
                max_prob, hard_label = torch.max(uw_prob, dim=1)
                over_threshold = max_prob > threshold
                if over_threshold.any():
                    u_i = u_i.to(device)
                    sample_index = u_i[over_threshold].tolist()
                    pseudo_label = hard_label[over_threshold].tolist()
                    for i, l in zip(sample_index, pseudo_label):
                        learning_status[i] = l

            # unsupervised loss
            batch_threshold = torch.index_select(cls_thresholds, 0, hard_label)
            indicator = max_prob > batch_threshold

            lu = (criterion(us_pred, hard_label) * indicator).mean()
            total_loss += lu * unsupervised_weight

        # optimization
        optimizer.zero_grad()
        if amp_flag:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        scheduler.step()
        ema.update()
        global_step += 1

        # logging
        metric.update_prediction(xw_pred, y)
        logs['Ls'].update(ls.item())
        logs['Mask'].update(torch.mean(indicator.float()).item())
        if indicator.any():
            logs['Lu'].update(lu.item())

    Acc = metric.calc_accuracy()
    Ls = logs['Ls'].avg
    Lu = logs['Lu'].avg
    Mask = logs['Mask'].avg

    return Acc, Ls, Lu, Mask


def train_network(args):
    """Train a network."""
    if args.wandb:
        import wandb
    global global_step
    global_step = 0

    device = torch.device('cuda')
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()

    # model
    model = get_network(args.network, args.num_classes)
    if args.mode == 'resume':
        ckpt = torch.load(args.load_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        ema = EMA(model=model, decay=args.ema_decay)
        ema.shadow.load_state_dict(ckpt['ema'])
        start_iter = ckpt['iteration']
    else:
        start_iter = 0
        ema = EMA(model=model, decay=args.ema_decay, device=device)
    model.to(device)

    # mapping function of beta
    mapping = mapping_func(args.mapping)

    # criterion
    criterion = nn.CrossEntropyLoss(reduction='none')

    # optimizer
    optimizer, scheduler = get_optimizer(model=model,
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         nesterov=args.nesterov,
                                         weight_decay=args.weight_decay,
                                         iterations=args.iterations)
    if args.mode == 'resume':
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    # labeled, unlabeled and test data
    X, U, T = get_dataloaders(data=args.data,
                              num_X=args.num_X,
                              include_x_in_u=args.include_x_in_u,
                              augs=args.augs,
                              batch_size=args.batch_size,
                              mu=args.mu)

    # Number of unlabeled data
    N = len(U.dataset.indices)
    if args.mode == 'resume' and 'learning_status' in ckpt:
        learning_status = ckpt['learning_status']
    else:
        learning_status = [-1] * N

    n_iter = 1024
    for epoch in range(start_iter//n_iter, n_iter):
        Acc, Ls, Lu, Mask = train_step(model=model,
                                       ema=ema,
                                       X=X,
                                       U=U,
                                       N=N,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       num_classes=args.num_classes,
                                       threshold=args.threshold,
                                       unsupervised_weight=args.lu_weight,
                                       amp_flag=args.amp,
                                       scaler=scaler,
                                       learning_status=learning_status,
                                       mapping=mapping,
                                       criterion=criterion,
                                       device=device)

        test_Acc = evaluate_step(ema.shadow, T, device)

        print((f"{time.ctime()}: "
               f"Iteration: [{global_step}/{args.iterations}], "
               f"Ls: {Ls:1.4f}, Lu: {Lu:1.4f}, Mask: {Mask:1.4f}, "
               f"Acc(train/test): [{Acc:1.4f}/{test_Acc:1.4f}]"))

        check_point = {
                'state_dict': model.state_dict(),
                'ema': ema.shadow.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'args': args,
                'learning_status': learning_status,
                'iteration': global_step
                }
        torch.save(check_point, args.save_path / 'ckpt.pth')
        if epoch % 10 == 0:
            torch.save(check_point, args.save_path / f'ckpt_{global_step}.pth')

        if args.wandb:
            wandb.log(data={'Ls': Ls,
                            'Lu': Lu,
                            'Mask': Mask,
                            'Train Acc': Acc,
                            'Test Acc': test_Acc},
                      step=global_step)
