import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn

# ===========================
# Problem / Training Settings
# ===========================
DEVICE = "cpu"
DTYPE  = torch.float32

L  = 1.0               # Pipe length [m]
rho = 1000.0           # Fluid density [kg/m^3] (e.g., water)
mu  = 1.0e-3           # Dynamic viscosity [Pa*s]

p_in  = 2.0e5          # Inlet pressure [Pa]
p_out = 1.0e5          # Outlet pressure [Pa]

# Collocation / BC sample sizes
N_COL = 2000           # interior collocation points
N_BC  = 100            # boundary points for each boundary condition

# Training hyperparameters
LR       = 1e-3
EPOCHS   = 20000
PRINT_EVERY = 1000
SEED     = 1234

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========
# Geometry
# =========
# Define diameter profile D(x) and area A(x).
# Ensure positivity and smoothness.
def D_np(x):
    """
    Example:  nozzle with oscillating diameter.
    D(x) in meters. Keep D_min > 0.
    """
    D0   = 0.05  # 5 cm nominal diameter
    eps  = 0.1   # amplitude fraction (0 < eps < 1 to keep diameter positive)
    # Smooth variation using cosine (period = L)
    return D0 * (1.0 + eps * (1.0 + np.cos(4.0*np.pi*x/L)) / 2.0)

def D_torch(x):
    D0   = 0.05
    eps  = 0.1
    return D0 * (1.0 + eps * (1.0 + torch.cos(4.0*math.pi*x/L)) / 2.0)

def A_np(x):
    D = D_np(x)
    return 0.25*np.pi*D**2

def A_torch(x):
    D = D_torch(x)
    return 0.25*math.pi*D**2

# =============
# Neural Models
# =============
class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=2, width=64, depth=5, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
        # Xavier init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# PINN that outputs [u(x), p(x)]
model = MLP(in_dim=1, out_dim=2, width=128, depth=6).to(DEVICE).to(DTYPE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========
# Samplers
# =========
def sample_collocation(n):
    x = np.random.rand(n, 1)*L
    return torch.tensor(x, dtype=DTYPE, device=DEVICE, requires_grad=True)

def sample_boundary_points(n):
    x0 = np.zeros((n//2, 1))
    xL = np.ones((n - n//2, 1))*L
    return (
        torch.tensor(x0, dtype=DTYPE, device=DEVICE, requires_grad=True),
        torch.tensor(xL, dtype=DTYPE, device=DEVICE, requires_grad=True),
    )

# ============
# PINN Losses
# ============
def pinn_residuals(x):
    """
    Compute physics residuals at interior collocation points.
    Unknowns: u(x), p(x).
    Equations enforced:
        1) Continuity: d/dx (A(x) * u) = 0
        2) Momentum:   rho * u * du/dx + dp/dx + 32*mu*u / D(x)^2 = 0
    """
    x.requires_grad_(True)
    out = model(x)
    u   = out[:, :1]
    p   = out[:, 1:]

    # Geometry terms
    A = A_torch(x)
    D = D_torch(x)

    # Spatial derivatives via autograd
    # du/dx
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    # dp/dx
    dp_dx = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    # d/dx (A u)
    Au = A * u
    dAu_dx = torch.autograd.grad(Au, x, grad_outputs=torch.ones_like(Au), create_graph=True)[0]

    # Residuals
    cont_res = dAu_dx                                      # continuity
    mom_res  = rho * u * du_dx + dp_dx + 32.0*mu*u/(D**2)  # momentum (laminar wall shear)

    return cont_res, mom_res

def bc_residuals(n):
    """
    Pressure Dirichlet BCs:
        p(0) = p_in,  p(L) = p_out
    """
    x0, xL = sample_boundary_points(n)
    out0 = model(x0)
    outL = model(xL)
    p0 = out0[:, 1:]
    pL = outL[:, 1:]

    bc0 = p0 - p_in
    bcL = pL - p_out
    return bc0, bcL

# Optional: Flow rate BC instead of p(0)
def flowrate_bc(n, Q_target=None):
    if Q_target is None:
        return None
    x0, _ = sample_boundary_points(n)
    out0 = model(x0)
    u0   = out0[:, :1]
    A0   = A_torch(x0)
    Q0   = A0 * u0
    return Q0 - Q_target

# ======
# Train
# ======
def train(epochs=EPOCHS, lr=LR, use_Q_bc=False, Q_target=None, w_cont=1.0, w_mom=100, w_bc=1):
    global optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        optimizer.zero_grad()

        # Interior physics
        x_col = sample_collocation(N_COL)
        cont_res, mom_res = pinn_residuals(x_col)

        # Boundary conditions
        bc0, bcL = bc_residuals(N_BC)
        bc_loss = (bc0**2).mean() + (bcL**2).mean()

        # Optional flowrate constraint
        if use_Q_bc and Q_target is not None:
            fr = flowrate_bc(N_BC, Q_target=Q_target)
            fr_loss = (fr**2).mean()
        else:
            fr_loss = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)

        # Total loss
        loss = (
            w_cont * (cont_res**2).mean() +
            w_mom  * (mom_res**2).mean() +
            w_bc   * bc_loss +
            w_bc   * fr_loss
        )

        loss.backward()
        optimizer.step()

        if ep % PRINT_EVERY == 0 or ep == 1:
            with torch.no_grad():
                uLmean = model(torch.tensor([[L]], dtype=DTYPE)).squeeze()[0].item()
                p0     = model(torch.tensor([[0.0]], dtype=DTYPE)).squeeze()[1].item()
                pL     = model(torch.tensor([[L]], dtype=DTYPE)).squeeze()[1].item()
            print(f"Epoch {ep:6d} | Loss {loss.item():.4e} | u(L)~{uLmean:.4e} m/s | p(0)~{p0:.3e} Pa | p(L)~{pL:.3e} Pa")

# ========================
# Utility: Evaluate fields
# ========================
def evaluate_on_grid(nx=201):
    """Return x, u(x), p(x) on a uniform grid for post-processing."""
    x = torch.linspace(0.0, L, nx, dtype=DTYPE).view(-1,1)
    with torch.no_grad():
        out = model(x)
        u = out[:, :1]
        p = out[:, 1:]
        A = A_torch(x)
        D = D_torch(x)
        Q = A*u
    return x.squeeze().cpu().numpy(), u.squeeze().cpu().numpy(), p.squeeze().cpu().numpy(), Q.squeeze().cpu().numpy(), A.squeeze().cpu().numpy(), D.squeeze().cpu().numpy()

# ==============
# Main (example)
# ==============
if __name__ == "__main__":
    print("Training a PINN for laminar incompressible quasi-1D pipe/nozzle flow...")
    print("Tip: Increase EPOCHS and adjust weights for better convergence.")
    # Example: also enforce an approximate flow rate computed from Poiseuille-like estimate
    # at the inlet diameter to help conditioning (optional).
    # You can set use_Q_bc=False to rely solely on pressure BCs.
    #use_Q_bc = False
    use_Q_bc = True
    Q_target = None

    train(epochs=20000, lr=1e-3, use_Q_bc=use_Q_bc, Q_target=Q_target)

    # Sample outputs
    x,u,p,Q,A,D = evaluate_on_grid(nx=201)
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(x, u); plt.xlabel("x [m]"); plt.ylabel("u [m/s]"); plt.title("Velocity")
        plt.figure()
        plt.plot(x, p); plt.xlabel("x [m]"); plt.ylabel("p [Pa]"); plt.title("Pressure")
        plt.figure()
        plt.plot(x, Q); plt.xlabel("x [m]"); plt.ylabel("Q [m^3/s]"); plt.title("Volumetric Flow")
        plt.figure()
        plt.plot(x, D); plt.xlabel("x [m]"); plt.ylabel("D [m]"); plt.title("Diameter profile")
        plt.show()
    except Exception as e:
        print("Matplotlib plotting skipped:", e)
