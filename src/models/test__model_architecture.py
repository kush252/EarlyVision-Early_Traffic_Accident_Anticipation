import torch
from cnn_model import SimpleCNN
model = SimpleCNN(num_classes=2)
dummy = torch.randn(4, 3, 224, 224)
out = model(dummy)

print(out.shape)  # (4, 2)
