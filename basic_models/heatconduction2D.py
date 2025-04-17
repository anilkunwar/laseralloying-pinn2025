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
    def __init__(self, k, rho, C, Lx, Ly, T_max):
        super().__init__()
        self.alpha = k / (rho * C)  # Thermal diffusivity
        self.Lx = Lx
        self.Ly = Ly
        self.T_max = T_max
        
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
            SmoothSigmoid(slope=0.5)
        )
        
        # Initialize SmoothSigmoid scale to ensure output in [0, 1]
        self.net[7].scale.data.fill_(1.0)  # Correct index for SmoothSigmoid
        
        # Define temperature range
        self.T_min = 300.0
        self.T_max = 400.0

    def forward(self, x, y, t):
        x_norm = x / self.Lx
        y_norm = y / self.Ly
        t_norm = t / self.T_max
        scaled_output = self.net(torch.cat([x_norm, y_norm, t_norm], dim=1))
        # Inverse scale to 300–400 K
        return self.T_min + (self.T_max - self.T_min) * scaled_output

def laplacian(T, x, y):
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T),
                            create_graph=True, retain_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T),
                            create_graph=True, retain_graph=True)[0]
    
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x),
                             create_graph=True, retain_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y),
                             create_graph=True, retain_graph=True)[0]
    return T_xx + T_yy

def physics_loss(model, x, y, t):
    T_pred = model(x, y, t)
    T_t = torch.autograd.grad(T_pred, t, grad_outputs=torch.ones_like(T_pred),
                             create_graph=True, retain_graph=True)[0]
    lap_T = laplacian(T_pred, x, y)
    residual = T_t - model.alpha * lap_T
    return torch.mean(residual**2)

def boundary_loss_bottom(model):
    num = 100
    x = torch.rand(num, 1) * model.Lx
    y = torch.zeros(num, 1)
    t = torch.rand(num, 1) * model.T_max
    T_pred = model(x, y, t)
    return torch.mean((T_pred - 400.0)**2)

def boundary_loss_top(model):
    num = 100
    x = torch.rand(num, 1) * model.Lx
    y = torch.full((num, 1), model.Ly)
    t = torch.rand(num, 1) * model.T_max
    T_pred = model(x, y, t)
    return torch.mean((T_pred - 300.0)**2)

def boundary_loss_sides(model):
    num = 100
    x_left = torch.zeros(num, 1, dtype=torch.float32, requires_grad=True)
    y_left = torch.rand(num, 1) * model.Ly
    t_left = torch.rand(num, 1) * model.T_max
    T_left = model(x_left, y_left, t_left)
    
    x_right = torch.full((num, 1), float(model.Lx), 
                        dtype=torch.float32, requires_grad=True)
    T_right = model(x_right, y_left, t_left)
    
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

def initial_loss(model):
    num = 500
    x = torch.rand(num, 1) * model.Lx
    y = torch.rand(num, 1) * model.Ly
    t = torch.zeros(num, 1)
    T_pred = model(x, y, t)
    return torch.mean((T_pred - 300.0)**2)

@st.cache_resource
def train_PINN(k, rho, C, Lx=100, Ly=100, T_max=10, epochs=5000, lr=0.001):
    model = TemperaturePINN(k, rho, C, Lx, Ly, T_max)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    x_pde = torch.rand(1000, 1, requires_grad=True) * Lx
    y_pde = torch.rand(1000, 1, requires_grad=True) * Ly
    t_pde = torch.rand(1000, 1, requires_grad=True) * T_max
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        phys_loss = physics_loss(model, x_pde, y_pde, t_pde)
        bnd_loss_bottom = boundary_loss_bottom(model)
        bnd_loss_top = boundary_loss_top(model)
        side_loss = boundary_loss_sides(model)
        init_loss = initial_loss(model)
        
        loss = 10 * phys_loss + 1000 * (bnd_loss_bottom + bnd_loss_top) + 50 * side_loss + 100 * init_loss
        
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Total Loss: {loss.item():.6f}, "
                  f"Physics: {phys_loss.item():.6f}, "
                  f"Bottom: {bnd_loss_bottom.item():.6f}, "
                  f"Top: {bnd_loss_top.item():.6f}, "
                  f"Side: {side_loss.item():.6f}, "
                  f"Initial: {init_loss.item():.6f}")
        
    return model

@st.cache_data
def evaluate_model(_model, times, Lx=100, Ly=100):
    x = torch.linspace(0, Lx, 100)
    y = torch.linspace(0, Ly, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    T_preds = []
    boundary_checks = {
        'T_bottom': [], 'T_top': []
    }
    
    for t_val in times:
        t = torch.full((X.numel(), 1), t_val)
        T_pred = _model(X.reshape(-1,1), Y.reshape(-1,1), t)
        T = T_pred.detach().numpy().reshape(100,100).T
        T_preds.append(T)
        st.write(f"Time {t_val:.2f}s: Temperature range = {np.min(T):.2f} K to {np.max(T):.2f} K")
        
        if t_val == times[0]:
            boundary_checks['T_bottom'] = T[:,0]   # y=0
            boundary_checks['T_top'] = T[:,-1]     # y=Ly
            st.write(f"T at y=0, x=[0, 50, 100]: {T[0,0]:.2f}, {T[50,0]:.2f}, {T[-1,0]:.2f} K")
    
    return X.numpy(), Y.numpy(), T_preds, boundary_checks

def create_plotly_plot(X, Y, T_list, times, Lx, Ly):
    st.markdown("## Temperature Dynamics Visualization")
    
    x_coords = X[:,0]
    y_coords = Y[0,:]
    
    global_max_T = max(np.max(T) for T in T_list)
    global_min_T = min(np.min(T) for T in T_list)
    
    time_index = st.slider(
        "Select Time Instance (seconds)",
        min_value=0,
        max_value=len(times)-1,
        value=0,
        format="%.2f s"
    )
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Contour(
            z=T_list[time_index],
            x=x_coords,
            y=y_coords,
            colorscale='Inferno',
            zmin=300.0,
            zmax=400.0,
            colorbar=dict(title='Temperature (K)'),
            hovertemplate=(
                'Temperature: %{z:.2f} K<br>'
                'X: %{x:.2f} μm<br>'
                'Y: %{y:.2f} μm'
            )
        )
    )
    
    fig.update_layout(
        height=600,
        width=800,
        margin=dict(l=50, r=50, t=80, b=50),
        xaxis_title="X (μm)",
        yaxis_title="Y (μm)",
        title=f"Temperature @ t={times[time_index]:.2f}s<br>Spatial Domain: {Lx:.0f}μm × {Ly:.0f}μm",
        hovermode='closest'
    )
    
    fig.update_yaxes(autorange=True)
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    st.title("Temperature PINN Visualization")
    
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
        T_max = st.number_input("Simulation Time [s]",
                              min_value=1.0, max_value=1000.0,
                              value=10.0, step=10.0)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        k = st.number_input("Thermal Conductivity (W/m·K)",
                           min_value=0.1, max_value=1000.0,
                           value=401.0, step=10.0)  # Example: Copper
    with col5:
        rho = st.number_input("Density (kg/m³)",
                             min_value=100.0, max_value=20000.0,
                             value=8960.0, step=100.0)  # Example: Copper
    with col6:
        C = st.number_input("Specific Heat (J/kg·K)",
                           min_value=10.0, max_value=2000.0,
                           value=385.0, step=10.0)  # Example: Copper

    times = np.linspace(0, T_max, 20)

    if st.button("Train Model"):
        evaluate_model.clear()
        with st.spinner(f"Training PINN for {Lx:.0f}μm × {Ly:.0f}μm domain..."):
            model = train_PINN(k, rho, C, Lx, Ly, T_max)
            st.session_state.model = model
            
    if 'model' in st.session_state:
        X, Y, T_preds, boundary_checks = evaluate_model(
            st.session_state.model, 
            times,
            Lx=Lx,
            Ly=Ly
        )
        
        st.subheader("Boundary Condition Checks (at t=0)")
        st.write("Temperature at bottom (y=0): Mean = {:.2f}K, Std = {:.2f}K, Expected ≈ 400K".format(
            np.mean(boundary_checks['T_bottom']), np.std(boundary_checks['T_bottom'])))
        st.write("Temperature at top (y=Ly): Mean = {:.2f}K, Std = {:.2f}K, Expected ≈ 300K".format(
            np.mean(boundary_checks['T_top']), np.std(boundary_checks['T_top'])))
        
        create_plotly_plot(X, Y, T_preds, times, Lx, Ly)
