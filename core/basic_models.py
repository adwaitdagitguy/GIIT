import torch
import torch.nn as nn

class PhysicsMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity_block = nn.Linear(2, 8)
        self.gravity_block = nn.Linear(2, 8)
        self.combiner = nn.Linear(16, 1)

    def forward(self, x):
        t = x[:, 0:1]
        v0 = x[:, 1:2]
        g = x[:, 2:3]
        vt = torch.cat([v0, t], dim=1)
        gt2 = torch.cat([g, t**2], dim=1)
        v_term = self.velocity_block(vt)
        g_term = self.gravity_block(gt2)
        combined = torch.cat([v_term, g_term], dim=1)
        return self.combiner(combined)
