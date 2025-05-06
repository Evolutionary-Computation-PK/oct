import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import MixUp, CutMix

class BatchMixAugmentor:
    def __init__(self, num_classes: int, mode: str = 'none', alpha: float = 0.4):
        self.num_classes = num_classes
        self.mode = mode.lower()
        self.alpha = alpha

        if self.mode == 'mixup':
            self.augment = MixUp(num_classes=num_classes, alpha=alpha)
        elif self.mode == 'cutmix':
            self.augment = CutMix(num_classes=num_classes, alpha=alpha)
        elif self.mode == 'none':
            self.augment = None
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'mixup', 'cutmix', 'none'.")

    def __call__(self, images: torch.Tensor, labels: torch.Tensor):
        labels = labels.long()
        labels = F.one_hot(labels, num_classes=self.num_classes).float()

        if self.augment is not None:
            images, labels = self.augment(images, labels)

        return images, labels
