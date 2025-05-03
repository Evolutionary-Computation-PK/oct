import torchvision.transforms.functional as F

class AugmentedPreprocessor:

    def __init__(self, augmentation, preprocessor):
        self.augmentation = augmentation
        self.preprocessor = preprocessor

    def __call__(self, image):
        aug_img = self.augmentation(image)
        aug_img = F.to_pil_image(aug_img)
        return self.preprocessor(aug_img)
