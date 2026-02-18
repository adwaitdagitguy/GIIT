"""
FIXED VERSION - Simpler physics, better learning
Task 1: Create and train a simple MLP on synthetic physics data
"""

import torch
import torch.nn as nn
import numpy as np

print("="*60)
print("ALIFARAZ - PHASE 1: Model Training (FIXED)")
print("="*60)

# ============================================
# Generate Synthetic Physics Data (SIMPLIFIED)
# ============================================
def generate_projectile_data(n_samples=5000):
    """
    SIMPLE projectile motion: y = v0*t - 0.5*g*t²
    NO drag (makes learning easier for Phase 1)
    
    Inputs: [t, v0, g, angle]  (4 features, but angle doesn't affect height for now)
    Output: [y] (height)
    """
    np.random.seed(42)
    
    # Generate random inputs
    t = np.random.uniform(0, 2, n_samples)        # time 0-2 sec (shorter range)
    v0 = np.random.uniform(10, 30, n_samples)     # initial velocity 10-30 m/s (narrower range)
    g = np.ones(n_samples) * 9.81                 # gravity constant
    angle = np.random.uniform(0, 90, n_samples)   # angle (not used in simple physics)
    
    # SIMPLE physics: just y = v0*t - 0.5*g*t²
    # No drag, no angle effects - pure 1D projectile
    y = v0*t - 0.5*g*t**2
    
    # Stack into feature matrix
    X = np.column_stack([t, v0, g, angle])
    Y = y.reshape(-1, 1)
    
    return torch.FloatTensor(X), torch.FloatTensor(Y)

print("\nGENERATING SYNTHETIC DATA")
X_train, Y_train = generate_projectile_data(5000)
X_test, Y_test = generate_projectile_data(500)

print(f"Training data: {X_train.shape[0]} samples")
print(f"Test data: {X_test.shape[0]} samples")
print(f"Input features: [time, velocity, gravity, angle]")
print(f"Output: [height]")
print(f"Physics: y = v0*t - 0.5*g*t² (no drag)")

# ============================================
# Define Model Architecture (BIGGER)
# ============================================
class TinyMLP(nn.Module):
    """
    Slightly bigger network for better learning
    Input: 4 features → Hidden: 16 neurons → Output: 1 value
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 16)   # Increased from 8 to 16
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

print("\nMODEL ARCHITECTURE:")
model = TinyMLP()

# Count parameters
num_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
num_params = sum(p.numel() for p in model.parameters())

print(f"Number of layers: {num_layers}")
print(f"Number of parameters: {num_params}")
print(f"Architecture: Linear(4→16) → ReLU → Linear(16→1)")

# ============================================
# Train the Model (BETTER HYPERPARAMETERS)
# ============================================
print("\nMODEL TRAINING")

# FIXED: Lower learning rate, more epochs
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # Increased from 0.001
criterion = nn.MSELoss()

epochs = 3000  # Increased from 2000
print_every = 500

for epoch in range(epochs):
    # Forward pass
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, Y_train)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Print progress
    if epoch % print_every == 0 or epoch == epochs - 1:
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, Y_test)
        print(f"  Epoch {epoch:4d}: Train Loss={loss.item():.4f}, Test Loss={test_loss.item():.4f}")

# Final evaluation
model.eval()
with torch.no_grad():
    final_train_loss = criterion(model(X_train), Y_train).item()
    final_test_loss = criterion(model(X_test), Y_test).item()

print(f"\nTraining complete!")
print(f"  Final train loss: {final_train_loss:.4f}")
print(f"  Final test loss: {final_test_loss:.4f}")

# ============================================
# Save Everything
# ============================================
print("\nSAVING FILES")

# Save the complete model (architecture + weights)
torch.save(model, "tiny_mlp.pt")
print("  Saved: tiny_mlp.pt")

# Save test data for demonstration
torch.save(X_test[:10], "test_inputs.pt")
torch.save(Y_test[:10], "test_targets.pt")
print("  Saved: test_inputs.pt (10 samples)")
print("  Saved: test_targets.pt (10 samples)")

# ============================================
# Validation Check
# ============================================
print("\nVALIDATION CHECK")

with torch.no_grad():
    # Check multiple samples
    errors = []
    for i in range(min(3, len(X_test))):
        sample_input = X_test[i:i+1]
        sample_target = Y_test[i:i+1]
        sample_pred = model(sample_input)
        error = abs(sample_pred.item() - sample_target.item())
        errors.append(error)
        
        if i == 0:  # Print first one in detail
            print(f"\n  Sample prediction #{i+1}:")
            print(f"    Input: t={sample_input[0,0]:.2f}s, v0={sample_input[0,1]:.2f}m/s, g={sample_input[0,2]:.2f}")
            print(f"    Target:    {sample_target.item():8.2f} m")
            print(f"    Predicted: {sample_pred.item():8.2f} m")
            print(f"    Error:     {error:8.2f} m")

avg_error = sum(errors) / len(errors)
print(f"\n  Average error on {len(errors)} samples: {avg_error:.2f} m")

# Success criteria
if final_test_loss < 2.0 and avg_error < 3.0:
    print("\n--MODEL TRAINING SUCCESSFUL--")
    print("Ready for intervention testing.")

else:
    print("WARNING: Model quality seems low.")
    print(f"Test loss: {final_test_loss:.4f} (expected < 2.0)")
    print(f"Avg error: {avg_error:.4f} (expected < 3.0)")
    print("\nPossible fixes:")
    print("1. Increase epochs to 5000")
    print("2. Try learning rate 0.01 or 0.001")
    print("3. Check if you have GPU available")

# Print physics sanity check
print("\nCHECKING PHYSICS")
print("  For t=1s, v0=20m/s, g=9.81:")
print(f"    True physics: y = 20*1 - 0.5*9.81*1² = {20*1 - 0.5*9.81*1**2:.2f} m")

test_case = torch.FloatTensor([[1.0, 20.0, 9.81, 45.0]])
with torch.no_grad():
    prediction = model(test_case)
print(f"    Model predicts: {prediction.item():.2f} m")
print(f"    Error: {abs(prediction.item() - (20*1 - 0.5*9.81*1**2)):.2f} m")