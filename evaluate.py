"""Evaluation Code."""

import torch
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score
from network import get_network
from data import get_dataloaders


class Metric:
    """Performance metric."""

    def __init__(self):
        """Set label."""
        self.reset()

    def reset(self):
        """Reset."""
        self.predictions = []
        self.ground_truths = []

    def update_prediction(self, y_pred, y):
        """Update prediction and gt."""
        self.ground_truths += y.tolist()

        with torch.no_grad():
            y_prob = softmax(y_pred, dim=1)
        self.predictions += y_prob.tolist()

        return None

    def calc_accuracy(self):
        """Get accuracy."""
        y_pred = [prob.index(max(prob)) for prob in self.predictions]
        return accuracy_score(y_true=self.ground_truths,
                              y_pred=y_pred)


def evaluate_step(model, dataloader, device):
    """Evaluate a network."""
    metric = Metric()

    for x, y, _ in dataloader:
        with torch.no_grad():
            y_hat = model(x[0].to(device))
        metric.update_prediction(y_hat, y.to(device))
    test_acc = metric.calc_accuracy()

    return test_acc


def evaluate_network(args):
    """Evaluate a network."""
    device = torch.device('cuda')

    model = get_network(args.network, args.num_classes)
    ckpt = torch.load(args.load_path, map_location='cpu')
    model.load_state_dict(ckpt['ema'])
    model.eval()
    model.to(device)

    # labeled, unlabeled and test data
    _, _, T = get_dataloaders(data=args.data,
                              num_X=args.num_X,
                              include_x_in_u=args.include_x_in_u,
                              augs=args.augs,
                              batch_size=args.batch_size,
                              mu=args.mu)

    test_acc = evaluate_step(model, T, device)

    print(f"Model Performance: {test_acc:1.4f}")
    return None
