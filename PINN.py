import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from pyDOE import lhs

# Set random seeds for reproducibility
np.random.seed(1234)
tf.random.set_seed(1234)

# Domain parameters
x_min, x_max = 0.0, 1.0
N_b, N_c = 100, 100

# Generate data
def generate_data():
    # Boundary data
    x_i = np.zeros((N_b, 1), dtype=np.float32)
    T_i = np.zeros((N_b, 1), dtype=np.float32)
    x_o = np.ones((N_b, 1), dtype=np.float32)
    T_o = np.zeros((N_b, 1), dtype=np.float32)
    
    x_bnd = np.concatenate([x_i, x_o], axis=0)
    T_bnd = np.concatenate([T_i, T_o], axis=0)
    
    # Collocation points
    x_col = x_min + (x_max - x_min) * lhs(1, N_c).astype(np.float32)
    
    # Convert to TensorFlow tensors
    x_bnd = tf.convert_to_tensor(x_bnd, dtype=tf.float32)
    T_bnd = tf.convert_to_tensor(T_bnd, dtype=tf.float32)
    x_col = tf.convert_to_tensor(x_col, dtype=tf.float32)
    
    return x_col, x_bnd, T_bnd

x_col, x_bnd, T_bnd = generate_data()

# Neural network architecture
def create_network():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation='tanh', input_shape=(1,), kernel_initializer='glorot_normal'),
        tf.keras.layers.Dense(20, activation='tanh', kernel_initializer='glorot_normal'),
        tf.keras.layers.Dense(1, activation=None, kernel_initializer='glorot_normal')
    ])
    return model

# Create network
net = create_network()

# Define loss functions
def boundary_loss(model, x_bnd, T_bnd):
    T_pred = model(x_bnd)
    return tf.reduce_mean(tf.square(T_pred - T_bnd))

def pde_loss(model, x_col):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_col)
        T = model(x_col)
        dTdx = tape.gradient(T, x_col)
    d2Tdx2 = tape.gradient(dTdx, x_col)
    
    # PDE: T_xx + (30x - 4) = 0
    residual = d2Tdx2 + (30 * x_col - 4)
    return tf.reduce_mean(tf.square(residual))

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Training loop
loss_history = {'bc': [], 'pde': []}

def train_step():
    with tf.GradientTape() as tape:
        bc_loss_val = boundary_loss(net, x_bnd, T_bnd)
        pde_loss_val = pde_loss(net, x_col)
        total_loss = bc_loss_val + pde_loss_val
    
    gradients = tape.gradient(total_loss, net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, net.trainable_variables))
    
    loss_history['bc'].append(bc_loss_val.numpy())
    loss_history['pde'].append(pde_loss_val.numpy())
    
    return total_loss

# Training
start_time = time.time()
for epoch in range(500):
    loss = train_step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/500 - Total Loss: {loss.numpy():.3e}")

print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes")

# Plotting functions
def plot_results():
    x_test = np.linspace(x_min, x_max, 100, dtype=np.float32).reshape(-1, 1)
    T_pred = net(x_test).numpy()
    
    exact_solution = -5 * x_test**3 + 2 * x_test**2 + 3 * x_test
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_test, exact_solution, label="Exact Solution", linestyle="-")
    plt.plot(x_test, T_pred, label="Predicted Solution", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("T(x)")
    plt.legend()
    plt.show()

def plot_loss():
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history['bc'], label='Boundary Loss')
    plt.semilogy(loss_history['pde'], label='PDE Loss')
    plt.xlabel("Iterations")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.show()

# Show results
plot_loss()
plot_results()




#####################################################################################

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import grad
import time
from pyDOE import lhs

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set random seeds
torch.manual_seed(1234)
np.random.seed(1234)

# Domain parameters
x_min, x_max = 0.0, 1.0
ub = torch.tensor([x_max], dtype=torch.float32).to(device)
lb = torch.tensor([x_min], dtype=torch.float32).to(device)
N_b, N_c = 100, 100

# Generate data
def generate_data():
    # Boundary data
    x_i = torch.zeros((N_b, 1), dtype=torch.float32, device=device)
    T_i = torch.zeros((N_b, 1), dtype=torch.float32, device=device)
    x_o = torch.ones((N_b, 1), dtype=torch.float32, device=device)
    T_o = torch.zeros((N_b, 1), dtype=torch.float32, device=device)
    
    x_bnd = torch.cat([x_i, x_o], dim=0)
    T_bnd = torch.cat([T_i, T_o], dim=0)
    
    # Collocation points
    x_col = lb + (ub - lb) * torch.tensor(lhs(1, N_c), dtype=torch.float32, device=device)
    x_col.requires_grad_(True)
    
    return x_col, x_bnd, T_bnd

x_col, x_bnd, T_bnd = generate_data()

# Neural network architecture
def create_network():
    net = nn.Sequential(
        nn.Linear(1, 20),
        nn.Tanh(),
        nn.Linear(20, 20),
        nn.Tanh(),
        nn.Linear(20, 1)
    ).to(device)
    
    # Xavier initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    net.apply(init_weights)
    return net

# Create network and optimizer
net = create_network()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Loss tracking
loss_history = {'bc': [], 'pde': []}

# Define loss functions
def boundary_loss(network, x_bnd, T_bnd):
    T_pred = network(x_bnd)
    return torch.mean((T_pred - T_bnd)**2)

def pde_loss(network, x_col):
    x = x_col.clone().requires_grad_(True)
    T = network(x)
    
    # First derivatives
    dTdx = grad(T.sum(), x, create_graph=True)[0]
    # Second derivatives
    d2Tdx2 = grad(dTdx.sum(), x, create_graph=True)[0]
    
    # PDE: T_xx + (30x - 4) = 0
    residual = d2Tdx2 + (30 * x - 4)
    return torch.mean(residual**2)

# Training loop
start_time = time.time()
for epoch in range(500):
    optimizer.zero_grad()
    
    # Calculate losses
    loss_bc = boundary_loss(net, x_bnd, T_bnd)
    loss_pde = pde_loss(net, x_col)
    total_loss = loss_bc + loss_pde
    
    # Backpropagation
    total_loss.backward()
    optimizer.step()
    
    # Track losses
    loss_history['bc'].append(loss_bc.item())
    loss_history['pde'].append(loss_pde.item())
    
    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/500 - Total Loss: {total_loss.item():.3e}")

print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes")

# Plotting functions
def plot_results():
    x_test = torch.linspace(x_min, x_max, 100, dtype=torch.float32, device=device).unsqueeze(-1)
    with torch.no_grad():
        T_pred = net(x_test).cpu().numpy()
    
    x_np = x_test.cpu().numpy()
    exact_solution = -5 * x_np**3 + 2 * x_np**2 + 3 * x_np
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_np, exact_solution, label="Exact Solution", linestyle="-")
    plt.plot(x_np, T_pred, label="PINN Prediction", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("T(x)")
    plt.legend()
    plt.show()

def plot_loss():
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history['bc'], label='Boundary Loss')
    plt.semilogy(loss_history['pde'], label='PDE Loss')
    plt.xlabel("Iterations")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.show()

# Show results
plot_loss()
plot_results()
