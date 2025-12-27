import torch
import torch.nn as nn
import sys

# 1. Define Architecture (Must match train_final.py)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t], axis=1))

# 2. Load the Brain
try:
    model = torch.load('hypersonic_thermal_model.pth', map_location=torch.device('cpu'), weights_only=False)
    model.eval()
except FileNotFoundError:
    print("❌ Error: Model not found.")
    sys.exit()

print("--- AI BRAIN SCAN DIAGNOSTIC ---")
print("Checking predictions for the first 10 seconds...")
print(f"{'Time (s)':<10} | {'AI Prediction (K)':<20}")
print("-" * 35)

# 3. Check specific time points
for t_val in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
    x_in = torch.tensor([[0.0]]) # Surface
    t_in = torch.tensor([[float(t_val)]])
    
    with torch.no_grad():
        pred_norm = model(x_in, t_in).item()
    
    # Convert back to Real Temp
    pred_temp = (pred_norm * 1200.0) + 300.0
    
    print(f"{t_val:<10.1f} | {pred_temp:<20.1f}")