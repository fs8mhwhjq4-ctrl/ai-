import torch
import torch.nn as nn
import numpy as np

# 1. Define the Neural Network (The Brain)
# Input: Position (x) and Time (t)
# Output: Temperature (T)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x, t):
        # Concatenate x and t to pass into the network
        inputs = torch.cat([x, t], axis=1)
        return self.net(inputs)

# 2. Program the Physics (The "Physics Loss")
def physics_loss(model, x, t, alpha=0.01):
    # Enable gradient tracking for physics calculation
    x.requires_grad = True
    t.requires_grad = True
    
    # Predict Temperature
    T = model(x, t)
    
    # Calculate Derivatives (The Calculus part)
    # dT/dt (Change in temp over time)
    T_t = torch.autograd.grad(T, t, torch.ones_like(T), create_graph=True)[0]
    
    # dT/dx (Change in temp over depth)
    T_x = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    
    # d^2T/dx^2 (Curvature of temp profile)
    T_xx = torch.autograd.grad(T_x, x, torch.ones_like(T_x), create_graph=True)[0]
    
    # The Heat Equation Residual: dT/dt - alpha * d^2T/dx^2 = 0
    # If the model understands physics, this result should be 0.
    residual = T_t - alpha * T_xx
    
    return torch.mean(residual ** 2)

# --- Training Loop Simulation ---
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Generate some random training points (collocation points)
x_physics = torch.rand(1000, 1) # Random depths
t_physics = torch.rand(1000, 1) # Random times

# Train
for epoch in range(5000):
    optimizer.zero_grad()
    
    # Calculate how much we are violating the laws of physics
    loss = physics_loss(model, x_physics, t_physics)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss {loss.item()}")

# Save the brain for your Arduino loop
torch.save(model, 'hypersonic_thermal_model.pth')