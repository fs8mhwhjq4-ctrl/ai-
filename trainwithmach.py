import torch
import torch.nn as nn
import numpy as np

# --- 1. SETUP ---
k, alpha = 20.0, 0.01
T_min = 300.0
T_range = 2000.0 # Increased range because Mach 5 is HOT

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # INPUT CHANGE: 3 inputs (x, t, Mach) -> 50 neurons
        self.net = nn.Sequential(
            nn.Linear(3, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1),
            nn.Sigmoid() 
        )
    def forward(self, x, t, m):
        # Concatenate x, t, AND m
        return self.net(torch.cat([x, t, m], axis=1))

# --- 2. PHYSICS LOSS (Variable Mach) ---
def compute_loss(model, x, t, m, x_bound, t_bound, m_bound, x_init, t_init, m_init):
    # A. Internal Physics
    x.requires_grad = True
    t.requires_grad = True
    
    T_norm = model(x, t, m)
    T_real = T_norm * T_range + T_min
    
    T_t = torch.autograd.grad(T_real, t, torch.ones_like(T_real), create_graph=True)[0]
    T_x = torch.autograd.grad(T_real, x, torch.ones_like(T_real), create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, torch.ones_like(T_x), create_graph=True)[0]
    
    loss_pde = torch.mean((T_t - alpha * T_xx) ** 2)

    # B. Boundary Condition (Variable Aero Heating!)
    x_bound.requires_grad = True
    t_bound.requires_grad = True
    # m_bound doesn't need grad, it's just a parameter
    
    T_surf_norm = model(x_bound, t_bound, m_bound)
    T_surf_real = T_surf_norm * T_range + T_min
    
    T_x_surf = torch.autograd.grad(T_surf_real, x_bound, torch.ones_like(T_surf_real), create_graph=True)[0]
    
    # --- DYNAMIC PHYSICS CALCULATIONS ---
    # 1. Recovery Temp increases with Mach^2
    # Approx: 300K at Mach 0, ~1500K at Mach 5
    T_recovery_dynamic = 300.0 + (50.0 * (m_bound ** 2))
    
    # 2. Heat Transfer Coeff (h) increases with Speed
    # Approx: Stagnant air (10) to Intense Aero (500)
    h_dynamic = 10.0 + (100.0 * m_bound)
    
    heat_flux_residual = -k * T_x_surf - h_dynamic * (T_recovery_dynamic - T_surf_real)
    
    # Scale down
    loss_bc = torch.mean((heat_flux_residual / 100000.0) ** 2)
    
    # C. Initial Condition (Start at 300K)
    T_init_norm = model(x_init, t_init, m_init)
    loss_ic = torch.mean((T_init_norm - 0.0) ** 2) * 1000.0

    return loss_pde + loss_bc + loss_ic

# --- 3. TRAINING LOOP ---
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Generate Data
# Mach numbers randomly between 0.0 and 5.0
x_phys = torch.rand(5000, 1)
t_phys = torch.rand(5000, 1) * 20.0
m_phys = torch.rand(5000, 1) * 5.0 

x_bound = torch.zeros(2000, 1)
t_bound = torch.rand(2000, 1) * 20.0
m_bound = torch.rand(2000, 1) * 5.0

x_init = torch.rand(1000, 1)
t_init = torch.zeros(1000, 1)
m_init = torch.rand(1000, 1) * 5.0

print("Training Variable Mach Model...")
for epoch in range(8000):
    optimizer.zero_grad()
    loss = compute_loss(model, x_phys, t_phys, m_phys, x_bound, t_bound, m_bound, x_init, t_init, m_init)
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.6f}")

torch.save(model, 'hypersonic_mach_model.pth')
print("✅ Mach-Capable Brain Saved!")