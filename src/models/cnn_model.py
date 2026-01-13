import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        # -------- Convolutional backbone --------
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)  # halves H, W

        # -------- Projection head --------
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, 3, 224, 224)

        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 56, 56)
        x = self.pool(F.relu(self.conv3(x)))  # (B, 128, 28, 28)

        x = self.gap(x)                       # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)             # (B, 128)

        x = self.fc(x)                        # (B, num_classes)
        return x


