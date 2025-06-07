import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import logging
import functools

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, image_size=224):
        self.image_size = image_size
        logger.debug(f"Initializing Preprocessor with image_size={image_size}")

        # Basic preprocessing for validation/test - cached
        self.basic_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB first
            transforms.ToTensor(),  # Then convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Augmentation for training - simplified but effective
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB first
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),  # Then convert to tensor
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

    @functools.lru_cache(maxsize=1000)
    def _cached_basic_transform(self, image_bytes):
        """Cache basic transformations (resize, normalize)"""
        # Convert bytes back to numpy array
        image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(self.image_size, self.image_size)
        return self.basic_transform(image)

    def preprocess(self, images, is_training=False):
        """
        Preprocess images for training or validation
        
        Args:
            images: numpy array or torch tensor of shape (N, H, W) or (H, W)
            is_training: whether to apply augmentation
            
        Returns:
            torch tensor of shape (N, 3, H, W) or (3, H, W)
        """
        start_time = time.time()
        
        # Handle single image case
        if len(images.shape) == 2:
            images = np.expand_dims(images, axis=0)
            single_image = True
            logger.debug("Single image case")
        else:
            single_image = False
            logger.debug("Multiple image case")

        self.validate_input(images)
        logger.debug(f"Input validation took {time.time() - start_time:.4f}s")

        # Convert to numpy if tensor
        if isinstance(images, torch.Tensor):
            images = images.numpy()
            logger.debug("Converted tensor to numpy")

        # Process images in batch
        transform = self.train_transform if is_training else self.basic_transform
        transform_start = time.time()
        
        # Process each image through the transform pipeline
        processed_images = []
        for img in images:
            if not is_training:
                # Cache basic transformations for validation
                img_bytes = img.tobytes()
                processed_img = self._cached_basic_transform(img_bytes)
            else:
                # No caching for training due to augmentations
                processed_img = transform(img)
            processed_images.append(processed_img)
        
        logger.debug(f"Transform application took {time.time() - transform_start:.4f}s")
        
        # Stack all processed images
        stack_start = time.time()
        result = torch.stack(processed_images)
        logger.debug(f"Stacking took {time.time() - stack_start:.4f}s")

        total_time = time.time() - start_time
        logger.debug(f"Total preprocessing took {total_time:.4f}s for {len(images)} images")
        
        # Return single image if input was single image
        if single_image:
            return result.squeeze(0)
        return result
