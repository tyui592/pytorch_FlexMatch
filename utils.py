"""Miscellaneous Code."""

import pickle


def save_pickle(data, path):
    """Save a dict to file."""
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return None


def load_pickle(path):
    """Load a pickle file."""
    with open(path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


class AverageMeter:
    """AverageMeter."""

    def __init__(self):
        """Set zero value."""
        self.reset()

    def reset(self):
        """Reset."""
        self.avg = 0
        self.sum = 0
        self.count = 0

        return None

    def update(self, val, n=1):
        """Update value."""
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        return None


def mapping_func(mode='convex'):
    """Returnt he Mapping func."""
    if mode == 'convex':
        return lambda x: x / (2 - x)
