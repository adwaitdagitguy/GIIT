import torch
import torch.nn as nn
import torch
import torch.nn as nn

# ---- Model definition (MUST match create_model.py) ----
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

model = torch.load("tiny_mlp.pt", weights_only=False)
model.eval()

print("Model loaded")

# Count layers
num_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
num_params = sum(p.numel() for p in model.parameters())

print("Number of layers:", num_layers)
print("Number of parameters:", num_params)

mapping = {
    "node1": "layer1",
    "node2": "layer2"
}
print("Mapping:", mapping)
input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
target = torch.tensor([[0.5]])

criterion = nn.MSELoss()

with torch.no_grad():
    baseline_output = model(input_tensor)
    baseline_metric = criterion(baseline_output, target)

print("Baseline MSE:", baseline_metric.item())

layer_to_mask = mapping["node1"]  # mask layer1

layer = getattr(model, layer_to_mask)

with torch.no_grad():
    layer.weight.zero_()
    if layer.bias is not None:
        layer.bias.zero_()

print(f"Masked {layer_to_mask}")

with torch.no_grad():
    intervened_output = model(input_tensor)
    intervened_metric = criterion(intervened_output, target)

print("Intervened MSE:", intervened_metric.item())

print("\nRESULTS")
print("Baseline Metric (before masking):", baseline_metric.item())
print("Intervened Metric (after masking):", intervened_metric.item())

