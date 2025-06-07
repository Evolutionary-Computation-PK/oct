import torch
import torch.nn as nn
from torchvision import models

from preprocessing.BatchMixAugmentor import BatchMixAugmentor
from preprocessing.DataLoaderFactory import DataLoaderFactory


class Classifier(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


if __name__ == '__main__':
    pre_args = {'image_size': 224, 'apply_blur': False, 'apply_clahe': True}
    aug_args = {'rotation_range': 10, 'zoom_range': 0.2, 'brightness_jitter': 0.1}
    factory = DataLoaderFactory(preprocessor_args=pre_args, augmentation_args=aug_args)
    train_loader, val_loader = factory.get_loaders()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    mix_mode = 'mixup'
    augmentor = BatchMixAugmentor(num_classes=4, mode=mix_mode, alpha=0.4)


    def soft_cross_entropy(preds, targets):
        log_probs = torch.nn.functional.log_softmax(preds, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()


    for epoch in range(1, 4):
        model.train()
        running_loss = 0.0

        for idx, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, labels = augmentor(imgs, labels)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = soft_cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch} complete. Avg Loss: {running_loss / len(train_loader):.4f}")
