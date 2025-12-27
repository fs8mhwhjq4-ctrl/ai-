import torch
import torch.nn as nn
import time
import sys

# 1. ARCHITECTURE (Must match 3 inputs)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1),
            nn.Sigmoid() 
        )
    def forward(self, x, t, m):
        return self.net(torch.cat([x, t, m], axis=1))

# 2. LOAD
try:
    model = torch.load('hypersonic_mach_model.pth', map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    print("✅ Loaded Mach Brain successfully!")
except FileNotFoundError:
    print("❌ Error: Run train_mach.py first!")
    sys.exit()

# 3. SIMULATION (With Physics Engine)
class HypersonicVehicle:
    def __init__(self):
        self.temp = 300.0
        self.mach = 0.0
        self.cooling_active = False
    
    def update(self, dt, target_mach):
        # 1. Accelerate/Decelerate
        if self.mach < target_mach:
            self.mach += 0.5 * dt 
        else:
            self.mach -= 0.5 * dt 
            if self.mach < 0: self.mach = 0.0
            
        # 2. Dynamic Aerodynamics (MATCHING THE AI TRAINING)
        # Calculate Recovery Temp (How hot the air is)
        T_recovery = 300.0 + (50.0 * (self.mach ** 2))
        
        # Calculate Heat Transfer Coefficient (How fast heat enters)
        # In training, we said h = 10 + 100 * M. 
        # We scale this by 0.0005 to fit the 0.1s time step physics.
        h_dynamic = 10.0 + (100.0 * self.mach)
        transfer_rate = h_dynamic * 0.0005 
        
        # Heat moves from Air to Skin
        delta_T = T_recovery - self.temp
        heating = delta_T * transfer_rate
        
        # 3. Apply Cooling
        cooling = 0.0
        if self.cooling_active and self.temp > 300.0:
            cooling = 800.0 * dt # Stronger cooling to fight the stronger heat!
            
        self.temp += heating - cooling
        
        # Physics Limits
        if self.temp < 300.0: self.temp = 300.0
        
        return self.temp, self.mach

jet = HypersonicVehicle()
start_time = time.time()
flight_phase = "ACCELERATING"
target_speed = 5.0

print("\n--- VARIABLE MACH FLIGHT TEST ---")
print("Phase 1: Accelerate to Mach 5")
print("Phase 2: Decelerate to Mach 0\n")

try:
    while True:
        current_time = time.time() - start_time
        
        # Flight Profile Logic
        if current_time > 10.0: 
            flight_phase = "DECELERATING"
            target_speed = 0.0
        
        current_temp, current_mach = jet.update(0.1, target_speed)

        # --- AI PREDICTION ---
        # Look 2 seconds ahead
        future_time = current_time + 2.0
        if future_time > 20.0: future_time = 20.0 # Cap time for AI stability
        
        # PREPARE INPUTS (Now including Mach!)
        x_in = torch.tensor([[0.0]])
        t_in = torch.tensor([[future_time]])
        m_in = torch.tensor([[current_mach]]) # The AI sees the speed
        
        with torch.no_grad():
            pred_norm = model(x_in, t_in, m_in).item()
            
        pred_temp = (pred_norm * 2000.0) + 300.0

        # CONTROL LOGIC
        status = "   "
        if pred_temp > 800.0:
            jet.cooling_active = True
            status = "❄️ COOLING"
        else:
            jet.cooling_active = False
            status = "🔥 HEATING"

        # DASHBOARD
        sys.stdout.write(f"\rT: {current_time:4.1f}s | Mach: {current_mach:3.1f} | Curr: {current_temp:5.0f} K | Pred: {pred_temp:5.0f} K | {status}")
        sys.stdout.flush()
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nTest Complete.")