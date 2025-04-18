import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SmoothSigmoid(nn.Module):
    def __init__(self, slope=1.0):
        super().__init__()
        self.k = slope
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.scale * 1 / (1 + torch.exp(-self.k * x))

class TemperaturePINN(nn.Module):
    def __init__(self, k, rho, C, Lx, Ly, Lz, T_max):
        super().__init__()
        self.alpha = k / (rho * C)
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.T_max = T_max
        self.rho = rho
        self.C = C
        
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
            SmoothSigmoid(slope=0.5)
        )
        
        self.net[7].scale.data.fill_(1.0)
        
        self.T_min = 300.0
        self.T_max = 900.0
        
        # Gaussian source parameters
        self.Q = 1e14 # W/m^3
        self.sigma = 0.5 * Lx  # μm
        self.v = Lx / T_max  # Velocity: Lx/T_max μm/s

    def forward(self, x, y, z, t):
        x_norm = x / self.Lx
        y_norm = y / self.Ly
        z_norm = z / self.Lz
        t_norm = t / self.T_max
        inputs = torch.cat([x_norm, y_norm, z_norm, t_norm], dim=1)
        scaled_output = self.net(inputs)
        return self.T_min + (self.T_max - self.T_min) * scaled_output

    def gaussian_source(self, x, y, z, t):
        # Gaussian source moving in x-direction: x(t) = v * t
        sigma_x = self.sigma
        sigma_y = self.sigma
        sigma_z = self.sigma
        x0 = self.v * t  # Moving center
        y0 = self.Ly / 2
        z0 = self.Lz
        source = self.Q * torch.exp(
            -((x - x0)**2) / (2 * sigma_x**2)
            - ((y - y0)**2) / (2 * sigma_y**2)
            - ((z - z0)**2) / (2 * sigma_z**2)
        )
        return source / (self.rho * self.C)

def laplacian(T, x, y, z):
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T),
                            create_graph=True, retain_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T),
                            create_graph=True, retain_graph=True)[0]
    T_z = torch.autograd.grad(T, z, grad_outputs=torch.ones_like(T),
                            create_graph=True, retain_graph=True)[0]
    
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x),
                             create_graph=True, retain_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y),
                             create_graph=True, retain_graph=True)[0]
    T_zz = torch.autograd.grad(T_z, z, grad_outputs=torch.ones_like(T_z),
                             create_graph=True, retain_graph=True)[0]
    return T_xx + T_yy + T_zz

def physics_loss(model, x, y, z, t):
    T_pred = model(x, y, z, t)
    T_t = torch.autograd.grad(T_pred, t, grad_outputs=torch.ones_like(T_pred),
                             create_graph=True, retain_graph=True)[0]
    lap_T = laplacian(T_pred, x, y, z)
    source = model.gaussian_source(x, y, z, t)
    residual = T_t - model.alpha * lap_T - source
    return torch.mean(residual**2)

def boundary_loss_bottom(model):
    num = 100
    x = torch.rand(num, 1) * model.Lx
    y = torch.rand(num, 1) * model.Ly
    z = torch.zeros(num, 1)
    t = torch.rand(num, 1) * model.T_max
    T_pred = model(x, y, z, t)
    return torch.mean((T_pred - 300.0)**2)

#def boundary_loss_top(model):
#    num = 100
#    x = torch.rand(num, 1) * model.Lx
#    y = torch.rand(num, 1) * model.Ly
#    z = torch.full((num, 1), model.Lz)
#   t = torch.rand(num, 1) * model.T_max
#    T_pred = model(x, y, z, t)
#    return torch.mean((T_pred - 300.0)**2)

# Revised boundary_loss_top for Neumann condition
# Replace the Dirichlet condition with a Neumann (zero flux) condition to allow temperature variation, when laser heat source is applied here
def boundary_loss_top(model):
    num = 100
    x = torch.rand(num, 1, requires_grad=True) * model.Lx
    y = torch.rand(num, 1, requires_grad=True) * model.Ly
    z = torch.full((num, 1), model.Lz, requires_grad=True)
    t = torch.rand(num, 1) * model.T_max
    T_pred = model(x, y, z, t)
    dT_dz = torch.autograd.grad(
        T_pred, z,
        grad_outputs=torch.ones_like(T_pred),
        create_graph=True, retain_graph=True
    )[0]
    return torch.mean(dT_dz**2)  # Minimize gradient (insulated)

def boundary_loss_x_sides(model):
    num = 100
    x_left = torch.zeros(num, 1, dtype=torch.float32, requires_grad=True)
    y = torch.rand(num, 1) * model.Ly
    z = torch.rand(num, 1) * model.Lz
    t = torch.rand(num, 1) * model.T_max
    T_left = model(x_left, y, z, t)
    
    x_right = torch.full((num, 1), float(model.Lx), 
                        dtype=torch.float32, requires_grad=True)
    T_right = model(x_right, y, z, t)
    
    grad_T_x_left = torch.autograd.grad(
        T_left, x_left,
        grad_outputs=torch.ones_like(T_left),
        create_graph=True, retain_graph=True
    )[0]
    
    grad_T_x_right = torch.autograd.grad(
        T_right, x_right,
        grad_outputs=torch.ones_like(T_right),
        create_graph=True, retain_graph=True
    )[0]
    
    return torch.mean(grad_T_x_left**2 + grad_T_x_right**2)

def boundary_loss_y_sides(model):
    num = 100
    x = torch.rand(num, 1) * model.Lx
    y_front = torch.zeros(num, 1, dtype=torch.float32, requires_grad=True)
    z = torch.rand(num, 1) * model.Lz
    t = torch.rand(num, 1) * model.T_max
    T_front = model(x, y_front, z, t)
    
    y_back = torch.full((num, 1), float(model.Ly), 
                       dtype=torch.float32, requires_grad=True)
    T_back = model(x, y_back, z, t)
    
    grad_T_y_front = torch.autograd.grad(
        T_front, y_front,
        grad_outputs=torch.ones_like(T_front),
        create_graph=True, retain_graph=True
    )[0]
    
    grad_T_y_back = torch.autograd.grad(
        T_back, y_back,
        grad_outputs=torch.ones_like(T_back),
        create_graph=True, retain_graph=True
    )[0]
    
    return torch.mean(grad_T_y_front**2 + grad_T_y_back**2)

def initial_loss(model):
    num = 500
    x = torch.rand(num, 1) * model.Lx
    y = torch.rand(num, 1) * model.Ly
    z = torch.rand(num, 1) * model.Lz
    t = torch.zeros(num, 1)
    T_pred = model(x, y, z, t)
    return torch.mean((T_pred - 300.0)**2)

@st.cache_resource
def train_PINN(k, rho, C, Lx=100, Ly=100, Lz=100, T_max=10, epochs=5000, lr=0.001):
    model = TemperaturePINN(k, rho, C, Lx, Ly, Lz, T_max)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    x_pde = torch.rand(1000, 1, requires_grad=True) * Lx
    y_pde = torch.rand(1000, 1, requires_grad=True) * Ly
    z_pde = torch.rand(1000, 1, requires_grad=True) * Lz
    t_pde = torch.rand(1000, 1, requires_grad=True) * T_max
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        phys_loss = physics_loss(model, x_pde, y_pde, z_pde, t_pde)
        bnd_loss_bottom = boundary_loss_bottom(model)
        bnd_loss_top = boundary_loss_top(model)
        x_side_loss = boundary_loss_x_sides(model)
        y_side_loss = boundary_loss_y_sides(model)
        init_loss = initial_loss(model)
        
        loss = 10 * phys_loss + 1000 * (bnd_loss_bottom + bnd_loss_top) + \
               50 * (x_side_loss + y_side_loss) + 100 * init_loss
        
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Total Loss: {loss.item():.6f}, "
                  f"Physics: {phys_loss.item():.6f}, "
                  f"Bottom: {bnd_loss_bottom.item():.6f}, "
                  f"Top: {bnd_loss_top.item():.6f}, "
                  f"X-Sides: {x_side_loss.item():.6f}, "
                  f"Y-Sides: {y_side_loss.item():.6f}, "
                  f"Initial: {init_loss.item():.6f}")
        
    return model

@st.cache_data
def evaluate_model(_model, times, Lx=100, Ly=100, Lz=100):
    n_points = 30
    x = torch.linspace(0, Lx, n_points)
    y = torch.linspace(0, Ly, n_points)
    z = torch.linspace(0, Lz, n_points)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    T_preds = []
    boundary_checks = {
        'T_bottom': [], 'T_top': [], 'T_source': []
    }
    
    for t_val in times:
        t = torch.full((X.numel(), 1), t_val)
        T_pred = _model(X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1), t)
        T = T_pred.detach().numpy().reshape(n_points, n_points, n_points)
        T_preds.append(T)
        st.write(f"Time {t_val:.2f}s: Temperature range = {np.min(T):.2f} K to {np.max(T):.2f} K")
        
        if t_val == times[0]:
            boundary_checks['T_bottom'] = T[:, :, 0]
            boundary_checks['T_top'] = T[:, :, -1]
            boundary_checks['T_source'] = T[0, n_points//2, -1]
            st.write(f"T at z=0, x=0, y=0: {T[0,0,0]:.2f} K, Expected ≈ 400K")
            st.write(f"T at z=Lz, x=0, y=Ly/2 (source, t=0): {T[0,n_points//2,-1]:.2f} K, Expected ≈ 300K")
        else:
            # Check temperature near source at current x(t) = v * t
            x_source = _model.v * t_val
            x_idx = min(int(x_source / Lx * n_points), n_points - 1)
            T_source = T[x_idx, n_points//2, -1]
            st.write(f"T at z=Lz, x={x_source:.2f}, y=Ly/2 (source, t={t_val:.2f}): {T_source:.2f} K, Expected > 300K")
    
    return X.numpy(), Y.numpy(), Z.numpy(), T_preds, boundary_checks

def create_plotly_plot(X, Y, Z, T_list, times, Lx, Ly, Lz):
    st.markdown("## 3D Temperature Dynamics with Moving Heat Source")
    
    x_coords = X[:,0,0]
    y_coords = Y[0,:,0]
    z_coords = Z[0,0,:]
    
    time_index = st.slider(
        "Select Time Instance (seconds)",
        min_value=0,
        max_value=len(times)-1,
        value=0,
        format="%.2f s"
    )
    
    t_val = times[time_index]
    source_x = (Lx / times[-1]) * t_val  # x(t) = v * t
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=T_list[time_index].flatten(),
            isomin=300.0,
            isomax=max(400.0, np.max(T_list[time_index])),
            opacity=0.3,
            surface_count=20,
            colorscale='Inferno',
            colorbar=dict(title='Temperature (K)'),
            hovertemplate=(
                'Temperature: %{value:.2f} K<br>'
                'X: %{x:.2f} μm<br>'
                'Y: %{y:.2f} μm<br>'
                'Z: %{z:.2f} μm'
            )
        )
    )
    
    fig.add_trace(
        go.Scatter3d(
            x=[source_x], y=[Ly/2], z=[Lz],
            mode='markers',
            marker=dict(size=5, color='blue'),
            name=f'Heat Source (t={t_val:.2f}s)'
        )
    )
    
    fig.update_layout(
        height=800,
        width=800,
        margin=dict(l=50, r=50, t=80, b=50),
        scene=dict(
            xaxis_title="X (μm)",
            yaxis_title="Y (μm)",
            zaxis_title="Z (μm)",
            aspectratio=dict(x=1, y=1, z=1)
        ),
        title=f"Temperature @ t={t_val:.2f}s<br>Domain: {Lx:.0f}μm × {Ly:.0f}μm × {Lz:.0f}μm"
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    st.title("3D Temperature PINN with Moving Gaussian Heat Source")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        Lx = st.number_input("Domain Length (X) [μm]", 
                           min_value=10.0, max_value=500.0, 
                           value=100.0, step=10.0)
    with col2:
        Ly = st.number_input("Domain Length (Y) [μm]",
                           min_value=10.0, max_value=500.0,
                           value=100.0, step=10.0)
    with col3:
        Lz = st.number_input("Domain Length (Z) [μm]",
                           min_value=10.0, max_value=500.0,
                           value=100.0, step=10.0)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        T_max = st.number_input("Simulation Time [s]",
                              min_value=1.0, max_value=1000.0,
                              value=10.0, step=10.0)
    with col5:
        k = st.number_input("Thermal Conductivity (W/m·K)",
                           min_value=0.1, max_value=1000.0,
                           value=401.0, step=10.0)
    with col6:
        rho = st.number_input("Density (kg/m³)",
                             min_value=100.0, max_value=20000.0,
                             value=8960.0, step=100.0)
    
    col7, col8 = st.columns(2)
    with col7:
        C = st.number_input("Specific Heat (J/kg·K)",
                           min_value=10.0, max_value=2000.0,
                           value=385.0, step=10.0)
    
    times = np.linspace(0, T_max, 10)

    if st.button("Train Model"):
        evaluate_model.clear()
        with st.spinner(f"Training PINN for {Lx:.0f}μm × {Ly:.0f}μm × {Lz:.0f}μm domain..."):
            model = train_PINN(k, rho, C, Lx, Ly, Lz, T_max)
            st.session_state.model = model
            
    if 'model' in st.session_state:
        X, Y, Z, T_preds, boundary_checks = evaluate_model(
            st.session_state.model, 
            times,
            Lx=Lx,
            Ly=Ly,
            Lz=Lz
        )
        
        st.subheader("Boundary and Source Checks")
        st.write("Temperature at bottom (z=0): Mean = {:.2f}K, Std = {:.2f}K, Expected ≈ 400K".format(
            np.mean(boundary_checks['T_bottom']), np.std(boundary_checks['T_bottom'])))
        st.write("Temperature at top (z=Lz): Mean = {:.2f}K, Std = {:.2f}K, Expected ≈ 300K".format(
            np.mean(boundary_checks['T_top']), np.std(boundary_checks['T_top'])))
        
        create_plotly_plot(X, Y, Z, T_preds, times, Lx, Ly, Lz)
