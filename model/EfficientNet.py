import torch
from torch import nn
from torchvision import models


class EfficientNetOct(nn.Module):
    def __init__(self, num_classes: int = 4, dense_units: int = 512, dropout:float = 0.5):
        super().__init__()
        self.base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dense_units, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.base_model(x)

    def freeze_base(self) -> None:
        for p in self.base_model.features.parameters():
            p.requires_grad = False

    def unfreeze_top_layers(self, num_layers: int) -> None:
        total_blocks = len(self.base_model.features)
        for i in range(total_blocks - num_layers, total_blocks):
            for p in self.base_model.features[i].parameters():
                p.requires_grad = True