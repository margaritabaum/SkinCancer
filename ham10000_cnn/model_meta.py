"""CNN + patient metadata, in the lecture's explicit __init__ / forward style.

The image branch is the same 5-block CNN. The metadata branch is a small MLP over
age, sex and localization. The two are concatenated before the classifier, so the
network can learn things like "an acral lesion in a 30-year-old is rarely melanoma".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNWithMetadata(nn.Module):
    def __init__(self, num_meta, num_classes=7, dropout=0.4):
        super(CNNWithMetadata, self).__init__()

        # ---------------- image branch ----------------
        self.conv1a = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(32)

        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2b = nn.BatchNorm2d(64)

        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3b = nn.BatchNorm2d(128)

        self.conv4a = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4b = nn.BatchNorm2d(256)

        self.conv5a = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn5a = nn.BatchNorm2d(512)
        self.conv5b = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5b = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ---------------- metadata branch ----------------
        # BatchNorm1d on the input keeps the one-hot columns and the standardized
        # age on a comparable scale.
        self.meta_bn = nn.BatchNorm1d(num_meta)
        self.meta_fc1 = nn.Linear(num_meta, 64)
        self.meta_fc2 = nn.Linear(64, 64)
        self.meta_drop = nn.Dropout(p=0.2)

        # ---------------- joint classifier ----------------
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 + 64, num_classes)

    def forward(self, img, meta):
        # ---- image branch ----
        activation = F.relu(self.bn1a(self.conv1a(img)))
        activation = F.relu(self.bn1b(self.conv1b(activation)))
        activation = self.pool(activation)

        activation = F.relu(self.bn2a(self.conv2a(activation)))
        activation = F.relu(self.bn2b(self.conv2b(activation)))
        activation = self.pool(activation)

        activation = F.relu(self.bn3a(self.conv3a(activation)))
        activation = F.relu(self.bn3b(self.conv3b(activation)))
        activation = self.pool(activation)

        activation = F.relu(self.bn4a(self.conv4a(activation)))
        activation = F.relu(self.bn4b(self.conv4b(activation)))
        activation = self.pool(activation)

        activation = F.relu(self.bn5a(self.conv5a(activation)))
        activation = F.relu(self.bn5b(self.conv5b(activation)))
        activation = self.pool(activation)

        image_features = self.gap(activation).view(-1, 512)

        # ---- metadata branch ----
        meta_features = F.relu(self.meta_fc1(self.meta_bn(meta)))
        meta_features = self.meta_drop(meta_features)
        meta_features = F.relu(self.meta_fc2(meta_features))

        # ---- join and classify ----
        combined = torch.cat([image_features, meta_features], dim=1)
        return self.fc(self.drop(combined))


if __name__ == "__main__":
    model = CNNWithMetadata(num_meta=19)
    print("parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    img = torch.randn(4, 3, 128, 128)
    meta = torch.randn(4, 19)
    print("output shape:", tuple(model(img, meta).shape))
