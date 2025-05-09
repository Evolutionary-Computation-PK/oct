import torch
import platform
from torch.utils.data import DataLoader, Dataset
from preprocessing.Preprocessor import Preprocessor

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

class DataLoaderFactory:
    def __init__(self, batch_size=32, num_workers=None, device=None):
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set num_workers based on system capabilities and device
        if num_workers is None:
            if platform.system() == 'Windows':
                # On Windows, use a small number of workers to avoid issues
                self.num_workers = 0  # Start with 0 workers on Windows
            else:
                # On Unix systems, use more workers
                self.num_workers = min(4, torch.multiprocessing.cpu_count())
        else:
            self.num_workers = num_workers
            
        # Adjust num_workers for CPU-only systems
        if self.device.type == 'cpu':
            self.num_workers = min(self.num_workers, 2)  # Limit workers for CPU
            
        self.preprocessor = Preprocessor()

    def get_loaders(self, X_train, y_train, X_val, y_val, batch_size=None):
        """
        Create data loaders for training and validation sets.
        
        Args:
            X_train: Training images (numpy array)
            y_train: Training labels (numpy array)
            X_val: Validation images (numpy array)
            y_val: Validation labels (numpy array)
            batch_size: Optional batch size override
        """
        batch_size = batch_size or self.batch_size
        
        # Create datasets
        train_dataset = OCTDataset(X_train, y_train, self.preprocessor, is_training=True)
        val_dataset = OCTDataset(X_val, y_val, self.preprocessor, is_training=False)
        
        # Configure DataLoader settings based on device
        pin_memory = self.device.type == 'cuda'  # Only use pin_memory for GPU
        
        # Create data loaders with optimized settings for the current device
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory
        )

        return train_loader, val_loader

