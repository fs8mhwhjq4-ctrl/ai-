import torch
import torch.nn as nn
import time
import sys

# 1. ARCHITECTURE (Must match train_final.py)
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

# 2. LOAD BRAIN
try:
    model = torch.load('hypersonic_thermal_model.pth', map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    print("✅ Loaded Brain successfully!")
except FileNotFoundError:
    print("❌ Error: Run train_final.py first!")
    sys.exit()

# 3. SIMULATION
class VirtualVehicle:
    def __init__(self):
        self.temp = 300.0
        self.cooling_active = False
    
    def update(self, dt):
        heating = 150.0 * dt
        cooling = 0.0
        
        # We only allow cooling if temp is actually high enough to matter
        if self.cooling_active and self.temp > 300.0:
            cooling = 400.0 * dt
            
        self.temp += heating - cooling
        
        if self.temp < 300.0:
            self.temp = 300.0
        return self.temp

vehicle = VirtualVehicle()
start_time = time.time()

print("\n--- HYPERSONIC FLIGHT TEST ---")
print("Target Limit: 800 K\n")

try:
    while True:
        current_time = time.time() - start_time
        current_temp = vehicle.update(0.1)
        
        # --- REMOVED THE RESET LOGIC SO TIME CAN MOVE ---
        
        # PREDICTION
        # We cap the future look-ahead time at 20s so the AI doesn't crash on long runs
        future_time = current_time + 2.0
        if future_time > 20.0: future_time = 20.0
        
        x_in = torch.tensor([[0.0]])
        t_in = torch.tensor([[future_time]])
        
        with torch.no_grad():
            prediction_norm = model(x_in, t_in).item()
            
        # Convert 0-1 back to Real Temp
        predicted_temp = (prediction_norm * 1200.0) + 300.0

        # CONTROL
        status = "   "
        if predicted_temp > 800.0:
            vehicle.cooling_active = True
            status = "❄️ ACTIVE COOLING"
        else:
            vehicle.cooling_active = False
            status = "🔥 HEATING UP..."

        sys.stdout.write(f"\rTime: {current_time:4.1f}s | Curr: {current_temp:6.0f} K | Pred: {predicted_temp:6.0f} K | {status}")
        sys.stdout.flush()
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nTest Complete.")