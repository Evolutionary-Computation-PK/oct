import torch
from torch.utils.data import Dataset
import numpy as np

class OCTDataset(Dataset):
    """Dataset for OCT images with preprocessing"""

    def __init__(self, images, labels, preprocessor, is_training=False):
        self.images = images
        self.labels = labels.squeeze()  # Squeeze labels to 1D
        self.preprocessor = preprocessor
        self.is_training = is_training

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        processed_img = self.preprocessor.preprocess(img, is_training=self.is_training)
        return processed_img, self.labels[idx]
