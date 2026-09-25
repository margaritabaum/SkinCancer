"""SkinCNN written in the lecture's explicit __init__ / forward style.

Every layer is named in __init__; forward() spells out the order of operations.
This replaces the nn.Sequential block() helper in cell 36 of the notebook.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes=7, dropout=0.4):
        super(CNN, self).__init__()

        # ---- block 1: 128 -> 64 ----
        self.conv1a = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(32)

        # ---- block 2: 64 -> 32 ----
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2b = nn.BatchNorm2d(64)

        # ---- block 3: 32 -> 16 ----
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3b = nn.BatchNorm2d(128)

        # ---- block 4: 16 -> 8 ----
        self.conv4a = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4b = nn.BatchNorm2d(256)

        # ---- block 5: 8 -> 4 ----
        self.conv5a = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn5a = nn.BatchNorm2d(512)
        self.conv5b = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5b = nn.BatchNorm2d(512)

        # ---- head ----
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, img):
        # block 1
        activation = F.relu(self.bn1a(self.conv1a(img)))
        activation = F.relu(self.bn1b(self.conv1b(activation)))
        activation = self.pool(activation)

        # block 2
        activation = F.relu(self.bn2a(self.conv2a(activation)))
        activation = F.relu(self.bn2b(self.conv2b(activation)))
        activation = self.pool(activation)

        # block 3
        activation = F.relu(self.bn3a(self.conv3a(activation)))
        activation = F.relu(self.bn3b(self.conv3b(activation)))
        activation = self.pool(activation)

        # block 4
        activation = F.relu(self.bn4a(self.conv4a(activation)))
        activation = F.relu(self.bn4b(self.conv4b(activation)))
        activation = self.pool(activation)

        # block 5
        activation = F.relu(self.bn5a(self.conv5a(activation)))
        activation = F.relu(self.bn5b(self.conv5b(activation)))
        activation = self.pool(activation)

        # global average pooling -> (batch, 512, 1, 1) -> (batch, 512)
        pooled = self.gap(activation)
        flattened = pooled.view(-1, 512)

        # No softmax here, for the same reason the slide has no sigmoid:
        # nn.CrossEntropyLoss applies it internally and needs raw logits.
        return self.fc(self.drop(flattened))


if __name__ == "__main__":
    model = CNN()
    print("parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    dummy = torch.randn(4, 3, 128, 128)
    print("output shape:", tuple(model(dummy).shape))
