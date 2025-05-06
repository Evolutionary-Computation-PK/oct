from medmnist import OCTMNIST
from torch.utils.data import Dataset

class OCTDataset(Dataset):
    def __init__(self, split, transform=None, download=False):
        self.data = OCTMNIST(split=split, download=download)
        self.images = self.data.imgs
        self.labels = self.data.labels.squeeze()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = self.transform(image)
        return image, label
