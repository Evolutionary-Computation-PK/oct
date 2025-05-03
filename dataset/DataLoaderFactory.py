import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

from dataset.Augmentation import Augmentation
from dataset.AugmentedPreprocessor import AugmentedPreprocessor
from dataset.OCTDataset import OCTDataset
from dataset.Preprocessor import Preprocessor

class DataLoaderFactory:
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        train_size: int = 200,
        extra_ratio: float = 0.2,
        download: bool = False,
        preprocessor_args: dict = None,
        augmentation_args: dict = None
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.extra_ratio = extra_ratio
        self.download = download

        preprocessor_args = preprocessor_args or {'apply_blur': True, 'apply_clahe': True}
        augmentation_args = augmentation_args or {}

        self.preprocessor = Preprocessor(**preprocessor_args)
        self.augmentation = Augmentation(**augmentation_args)
        self.aug_preprocessor = AugmentedPreprocessor(self.augmentation, self.preprocessor)
        self.val_preprocessor = Preprocessor()

    def get_loaders(self):
        base_ds = OCTDataset(split='train', transform=self.preprocessor, download=self.download)
        total = len(base_ds)
        perm = torch.randperm(total)
        b_n = self.train_size
        e_n = int(b_n * self.extra_ratio)
        base_idx = perm[:b_n]
        extra_idx = perm[:e_n]

        base_subset = Subset(base_ds, base_idx)
        aug_ds = OCTDataset(split='train', transform=self.aug_preprocessor, download=self.download)
        extra_subset = Subset(aug_ds, extra_idx)

        train_ds = ConcatDataset([base_subset, extra_subset])
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        val_ds = OCTDataset(split='val', transform=self.val_preprocessor, download=self.download)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader

