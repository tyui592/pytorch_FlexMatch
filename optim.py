"""Optimization and Schduler Code."""

import math
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer(model, lr, momentum, nesterov, weight_decay, iterations):
    """Get optimizer and schduler."""
    def _cos_lr_decay(K):
        return lambda k: max(0, math.cos((7 * math.pi * k) / (16 * K)))

    optimizer = get_SGD(model, lr, momentum, weight_decay, nesterov)

    scheduler = LambdaLR(optimizer=optimizer,
                         lr_lambda=_cos_lr_decay(K=iterations))

    return optimizer, scheduler


def get_SGD(model, lr=0.1, momentum=0.9,
            weight_decay=5e-4, nesterov=True):
    """Return a optimizer."""
    no_decay = ['bias', 'bn']

    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = SGD(grouped_parameters,
                    lr=lr,
                    momentum=momentum,
                    nesterov=nesterov)
    return optimizer
