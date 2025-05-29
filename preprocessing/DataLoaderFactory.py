import torch
import platform
from torch.utils.data import DataLoader, Dataset
from preprocessing.Preprocessor import Preprocessor
import numpy as np
import multiprocessing as mp
import logging
import os

logger = logging.getLogger(__name__)

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
        return img, self.labels[idx]  # Return raw image and label


class DataLoaderFactory:
    def __init__(self, batch_size=32, num_workers=None, device=None):
        self.batch_size = batch_size
        
        # Enhanced GPU detection for Linux
        if platform.system() == 'Linux':
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"Found {gpu_count} NVIDIA GPU(s) on Linux system")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
                    logger.info(f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.1f}GB")
                self.device = device or torch.device('cuda')
            else:
                logger.warning("No NVIDIA GPUs found on Linux system, falling back to CPU")
                self.device = torch.device('cpu')
        else:
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Optimize num_workers based on system and GPU availability
        if num_workers is None:
            if platform.system() == 'Linux' and self.device.type == 'cuda':
                # For Linux with GPU, use more workers
                self.num_workers = min(4 * torch.cuda.device_count(), mp.cpu_count())
            elif platform.system() == 'Windows':
                self.num_workers = 0  # Disable multiprocessing on Windows
            else:
                self.num_workers = min(8, mp.cpu_count())
        else:
            self.num_workers = num_workers

        # Adjust num_workers for CPU-only systems
        if self.device.type == 'cpu':
            self.num_workers = min(self.num_workers, 4)

        self.preprocessor = Preprocessor()
        logger.info(f"Initialized DataLoaderFactory with batch_size={batch_size}, num_workers={self.num_workers}, device={self.device}")

    def collate_fn(self, batch, is_training=False):
        """
        Process a batch of images and labels.
        
        Args:
            batch: List of (image, label) tuples
            is_training: Whether to apply training augmentations
            
        Returns:
            Tuple of (processed_images, labels)
        """
        # Unzip the batch
        images, labels = zip(*batch)
        
        # Stack images into a batch
        images = np.stack(images)
        
        # Process the entire batch at once
        processed_images = self.preprocessor.preprocess(images, is_training=is_training)
        
        # Convert labels to tensor
        labels = torch.tensor(labels)
        
        return processed_images, labels

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
        logger.info(f"Creating data loaders with batch_size={batch_size}")

        # Create datasets
        train_dataset = OCTDataset(X_train, y_train, self.preprocessor, is_training=True)
        val_dataset = OCTDataset(X_val, y_val, self.preprocessor, is_training=False)

        # Configure DataLoader settings based on device
        pin_memory = self.device.type == 'cuda'
        
        # Set CUDA environment variables for better performance on Linux
        if platform.system() == 'Linux' and self.device.type == 'cuda':
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            os.environ['CUDA_CACHE_DISABLE'] = '0'

        # Create data loaders with optimized settings for the current device
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=self.num_workers > 0,  # Enable persistent workers if using multiple workers
            prefetch_factor=2 if self.num_workers > 0 else None,  # Enable prefetching for multiple workers
            generator=torch.Generator().manual_seed(42),
            collate_fn=lambda x: self.collate_fn(x, is_training=True)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=self.num_workers > 0,  # Enable persistent workers if using multiple workers
            prefetch_factor=2 if self.num_workers > 0 else None,  # Enable prefetching for multiple workers
            collate_fn=lambda x: self.collate_fn(x, is_training=False)
        )

        logger.info(f"Created train loader with {len(train_loader)} batches and val loader with {len(val_loader)} batches")
        return train_loader, val_loader
