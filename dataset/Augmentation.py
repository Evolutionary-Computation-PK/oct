import torchvision.transforms as transforms
from PIL import Image

class Augmentation:
    def __init__(
        self,
        rotation_range=15,               # Maximum rotation in degrees (±rotation_range)
        zoom_range=0.1,                  # Zoom range (scaling factor from 0.9 to 1.1)
        width_shift_range=0.1,           # Horizontal shift (as a fraction of image width)
        height_shift_range=0.1,          # Vertical shift (as a fraction of image height)
        brightness_jitter=0.2,           # Brightness variation factor
        horizontal_flip=True,            # Whether to apply horizontal flipping
        fill=0                           # Pixel fill value for empty areas (0 = black)
    ):
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=rotation_range, fill=fill),       # Random rotation within ±rotation_range, fill empty areas with 'fill'
            transforms.RandomAffine(                                           # Affine transform: shifting and zooming
                degrees=0,                                                     # No additional rotation
                translate=(width_shift_range, height_shift_range),            # Horizontal and vertical shift as a fraction of size
                scale=(1 - zoom_range, 1 + zoom_range),                        # Scaling range for zoom
                fill=fill                                                      # Fill value for uncovered areas
            ),
            transforms.ColorJitter(brightness=brightness_jitter),              # Random brightness adjustment
            transforms.RandomHorizontalFlip(p=0.5 if horizontal_flip else 0),  # Random horizontal flip with 50% probability (or 0%)
            transforms.Resize((28, 28)),                                       # Resize image to 28x28 (standard for OCTMNIST)
            transforms.ToTensor()                                              # Convert PIL image to PyTorch tensor
        ])

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return self.transform(image)
