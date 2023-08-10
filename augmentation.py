"""Augmentation Code."""

import torchvision.transforms as T
from randaugment import RandAugment


class Augmenter:
    """Image Augmenter."""

    def __init__(self, policy):
        """Get a policy."""
        # for model evaluation
        if policy == 0:
            self.transform = lambda x: x

        # weak augmentation
        elif policy == 1:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(32, padding=4, padding_mode='reflect')
            ])

        # strong augmentation
        elif policy == 2:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(32, padding=4, padding_mode='reflect'),
                RandAugment(n=2, m=10)
            ])

    def __call__(self, image):
        """Augmenter."""
        return self.transform(image)
