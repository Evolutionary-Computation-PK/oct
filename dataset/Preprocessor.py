import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

class Preprocessor:
    def __init__(self,
                 image_size=224,
                 apply_blur=False,
                 blur_ksize=(5, 5),
                 blur_sigma=0,
                 apply_clahe=False,
                 clahe_clip_limit=2.0,
                 clahe_tile_grid_size=(8, 8)):

        self.apply_blur = apply_blur
        self.blur_ksize = blur_ksize
        self.blur_sigma = blur_sigma

        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

        self.resize_shape = (image_size, image_size)

        self.torch_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resize_shape),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess_cv2(self, image: np.ndarray) -> np.ndarray:
        if self.apply_blur:
            image = cv2.GaussianBlur(image, self.blur_ksize, self.blur_sigma)

        if self.apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit,
                                    tileGridSize=self.clahe_tile_grid_size)
            image = clahe.apply(image)

        return image

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = self.preprocess_cv2(image)
        return self.torch_transforms(image)
