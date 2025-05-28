import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, image_size=224):
        self.image_size = image_size
        logger.debug(f"Initializing Preprocessor with image_size={image_size}")

        # Basic preprocessing for validation/test
        self.basic_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentation for training
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(brightness=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def validate_input(self, images):
        """Validate input images dimensions"""
        if not isinstance(images, (np.ndarray, torch.Tensor)):
            raise TypeError("Input must be numpy array or torch tensor")

        if len(images.shape) != 3:
            raise ValueError(f"Expected 3D input (N, H, W), got shape {images.shape}")

        if images.shape[1] != self.image_size or images.shape[2] != self.image_size:
            raise ValueError(
                f"Images must be {self.image_size}x{self.image_size}, got {images.shape[1]}x{images.shape[2]}")

    def preprocess(self, images, is_training=False):
        """
        Preprocess images for training or validation
        
        Args:
            images: numpy array or torch tensor of shape (N, H, W) or (H, W)
            is_training: whether to apply augmentation
            
        Returns:
            torch tensor of shape (N, 3, H, W) or (3, H, W)
        """
        # Handle single image case
        if len(images.shape) == 2:
            images = np.expand_dims(images, axis=0)
            single_image = True
        else:
            single_image = False

        self.validate_input(images)
        logger.debug(f"Input validation took {time.time() - start_time:.4f}s")

        # Convert to numpy if tensor
        if isinstance(images, torch.Tensor):
            images = images.numpy()
            logger.debug("Converted tensor to numpy")

        # Process each image
        processed_images = []
        transform = self.train_transform if is_training else self.basic_transform
        
        for img in images:
            processed_img = transform(img)
            processed_images.append(processed_img)
            
        result = torch.stack(processed_images)
        
        # Return single image if input was single image
        if single_image:
            return result.squeeze(0)
        return result
