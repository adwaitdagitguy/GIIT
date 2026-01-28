import torch
import torch.nn as nn

# Simple MLP
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = TinyMLP()

# Save model
torch.save(model, "tiny_mlp.pt")
print("Model saved as tiny_mlp.pt")
