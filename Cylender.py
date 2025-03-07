"""
PINN Method for the 2D Steady State Incompressible Navier-Stokes Equations
#EhsanGh94 
"""

# ! pip install pyDOE
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import grad
from pyDOE import lhs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import pickle ## Not Safe!!
# import random
# from torch.utils.data import TensorDataset, DataLoader


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(1234)
np.random.seed(1234)
# random.seed(1234)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(device)


x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 0.4
r = 0.1
xc = 0.4
yc = 0.2

T_wall = 1
T_in = 0.5
ub = np.array([x_max, y_max])
lb = np.array([x_min, y_min])

N_b = 200  # inlet & outlet
N_w = 400  # wall
N_s = 200  # surface
N_c = 40000  # collocation
N_r = 10000


def getData():

    inlet_xy = [x_min, y_min] + [0.0, y_max] * lhs(2, N_b)
    inlet_y = inlet_xy[:,1:2]
    inlet_x = np.zeros((N_b, 1))
    inlet_T = T_in*np.ones((N_b,1))

    inlet_u = 4 * inlet_y * (0.4 - inlet_y) / (0.4 ** 2) 
    inlet_v = np.zeros((N_b, 1))

    inlet_xy = np.concatenate([inlet_x, inlet_y], axis=1)
    inlet_uv = np.concatenate([inlet_u, inlet_v], axis=1)

    xy_outlet = [x_max, y_min] + [0.0, y_max] * lhs(2, N_b)  

    upwall_xy = [x_min, y_max] + [x_max, 0.0] * lhs(2, N_w)
    dnwall_xy = [x_min, y_min] + [x_max, 0.0] * lhs(2, N_w)

    upwall_uv = np.zeros((N_w, 2))
    dnwall_uv = np.zeros((N_w, 2))
    dnwall_T = np.ones((N_w,1))*T_wall
    upwall_T = np.ones((N_w,1))*T_wall

    # cylinder surface, u=v=0
    # theta = np.linspace(0.0, 2 * np.pi, N_s)
    theta = [0.0] + [2 * np.pi] * lhs(1, N_s)
    cyl_x = (r * np.cos(theta) + xc).reshape(-1, 1)
    cyl_y = (r * np.sin(theta) + yc).reshape(-1, 1)
    cyl_xy = np.concatenate([cyl_x, cyl_y], axis=1)
    cyl_uv = np.zeros((N_s, 2))
    cyl_T = np.ones((N_s, 1)) * T_wall

    xy_bnd = np.concatenate([inlet_xy, upwall_xy, dnwall_xy, cyl_xy], axis=0)
    uv_bnd = np.concatenate([inlet_uv, upwall_uv, dnwall_uv, cyl_uv], axis=0)
    T_bnd =  np.concatenate([inlet_T, upwall_T, dnwall_T, cyl_T], axis=0)


    xy_col = lb + (ub - lb) * lhs(2, N_c)

    # refine points around cylider
    refine_ub = np.array([xc + 2 * r, yc + 2 * r])
    refine_lb = np.array([xc - 2 * r, yc - 2 * r])
    xy_col_refine = refine_lb + (refine_ub - refine_lb) * lhs(2, N_r)
    xy_col = np.concatenate([xy_col, xy_col_refine], axis=0)

    # remove collocation points inside the cylinder
    dst_from_cyl = np.sqrt((xy_col[:, 0] - xc) ** 2 + (xy_col[:, 1] - yc) ** 2)
    xy_col = xy_col[dst_from_cyl > r].reshape(-1, 2)

    xy_col = np.concatenate((xy_col, xy_bnd, xy_outlet), axis=0)

    print(xy_col.shape)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    plt.scatter(xy_col[:,0:1], xy_col[:,1:2], marker='o', alpha=0.4 ,color='blue')
    plt.scatter(upwall_xy[:,0:1], upwall_xy[:,1:2], marker='o', alpha=0.5 , color='green')
    plt.scatter(dnwall_xy[:,0:1], dnwall_xy[:,1:2], marker='o', alpha=0.5 , color='green')
    plt.scatter(xy_outlet[:, 0:1], xy_outlet[:, 1:2], marker='o', alpha=0.5, color='orange')
    plt.scatter(inlet_xy[:, 0:1], inlet_xy[:, 1:2], marker='o', alpha=0.5, color='red')
    plt.show()

    xy_bnd = torch.tensor(xy_bnd, dtype=torch.float32).to(device)
    uv_bnd = torch.tensor(uv_bnd, dtype=torch.float32).to(device)
    T_bnd = torch.tensor(T_bnd, dtype=torch.float32).to(device)
    xy_outlet = torch.tensor(xy_outlet, dtype=torch.float32).to(device)
    xy_col = torch.tensor(xy_col, dtype=torch.float32).to(device)

    return xy_col, xy_bnd, uv_bnd,T_bnd , xy_outlet

xy_col, xy_bnd, uv_bnd ,T_bnd , xy_outlet = getData()

def plotLoss(losses_dict, path, info=["I.C.", "B.C.", "P.D.E."]):
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 6))
    axes[0].set_yscale("log")
    for i, j in zip(range(3), info):
        axes[i].plot(losses_dict[j.lower()])
        axes[i].set_title(j)
    plt.show()
    fig.savefig(path)

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

class layer(nn.Module):

    def __init__(self, n_in, n_out, activation):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x

class DNN(nn.Module):

    def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb, activation=nn.Tanh()): 
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(layer(dim_in, n_node, activation))   
        for _ in range(n_layer):
            self.net.append(layer(n_node, n_node, activation))
        self.net.append(layer(n_node, dim_out, activation=None))
        self.ub = torch.tensor(ub, dtype=torch.float).to(device)  
        self.lb = torch.tensor(lb, dtype=torch.float).to(device)
        self.net.apply(weights_init)  

    def forward(self, x):
        x = (x - self.lb) / (self.ub - self.lb)  
        out = x
        for layer in self.net:
            out = layer(out)
        return out

class PINN:

    # rho = 1
    # mu = 0.02
    # cp = 1000
    # landa = 0.03
    Re = 50
    Pr = 666.666
    def __init__(self) -> None:
        self.net = DNN(dim_in=2, dim_out=9, n_layer=4, n_node=50, ub=ub, lb=lb).to(
            device
        )

        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=20000,
            max_eval=20000,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps, 
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        self.adam = torch.optim.Adam(self.net.parameters(), lr=5e-4)
        self.losses = {"bc": [], "outlet": [], "pde": []}
        self.iter = 0

    def predict(self, xy):
        out = self.net(xy)

        u = out[:, 0:1]
        v = out[:, 1:2]
        T = out[:, 2:3]
        p = out[:, 3:4]


        sig_xx = out[:, 4:5]
        sig_xy = out[:, 5:6]
        sig_yy = out[:, 6:7]
        q_x = out[:, 7:8]
        q_y = out[:, 8:9]


        return u, v, T ,p, sig_xx, sig_xy, sig_yy,q_x,q_y

    def bc_loss(self, xy):
        u, v,T= self.predict(xy)[0:3]

        mse_bc = torch.mean(torch.square(u - uv_bnd[:, 0:1])) + torch.mean(
            torch.square(v - uv_bnd[:, 1:2]))+torch.mean(torch.square(T - T_bnd))

        return mse_bc
    
    def outlet_loss(self, xy):
        out = self.net(xy)

        p = out[:, 3:4]

        mse_outlet = torch.mean(torch.square(p))

        return mse_outlet
    
    def pde_loss(self, xy):
        xy = xy.clone()
        xy.requires_grad = True

        u, v, T, p, sig_xx, sig_xy, sig_yy ,q_x,q_y= self.predict(xy)

        u_out = grad(u.sum(), xy, create_graph=True)[0]
        v_out = grad(v.sum(), xy, create_graph=True)[0]
        T_out = grad(T.sum(), xy, create_graph=True)[0]

        sig_xx_out = grad(sig_xx.sum(), xy, create_graph=True)[0]
        sig_xy_out = grad(sig_xy.sum(), xy, create_graph=True)[0]
        sig_yy_out = grad(sig_yy.sum(), xy, create_graph=True)[0]
        q_x_out = grad(q_x.sum(), xy, create_graph=True)[0]
        q_y_out = grad(q_y.sum(), xy, create_graph=True)[0]

        u_x = u_out[:, 0:1]
        u_y = u_out[:, 1:2]
        v_x = v_out[:, 0:1]
        v_y = v_out[:, 1:2]
        T_x = T_out[:, 0:1]
        T_y = T_out[:, 1:2]

        sig_xx_x = sig_xx_out[:, 0:1]
        sig_xy_x = sig_xy_out[:, 0:1]
        sig_xy_y = sig_xy_out[:, 1:2]
        sig_yy_y = sig_yy_out[:, 1:2]
        q_x_x = q_x_out[:,0:1]
        q_y_y = q_y_out[:,1:2]
      
        f0 = u_x + v_y

        f1 = (u * u_x + v * u_y) - sig_xx_x - sig_xy_y
        f2 = (u * v_x + v * v_y) - sig_xy_x - sig_yy_y

        f3 = -p + (2 / self.Re) * u_x - sig_xx
        f4 = -p + (2 / self.Re) * v_y - sig_yy
        f5 = (1 / self.Re) * (u_y + v_x) - sig_xy
        f6 = (u * T_x + v * T_y) - (1 / (self.Re * self.Pr)) * (q_x_x + q_y_y)
        f7 = q_x - T_x
        f8 = q_y - T_y


        mse_f0 = torch.mean(torch.square(f0))
        mse_f1 = torch.mean(torch.square(f1))
        mse_f2 = torch.mean(torch.square(f2))
        mse_f3 = torch.mean(torch.square(f3))
        mse_f4 = torch.mean(torch.square(f4))
        mse_f5 = torch.mean(torch.square(f5))
        mse_f6 = torch.mean(torch.square(f6))
        mse_f7 = torch.mean(torch.square(f7))
        mse_f8 = torch.mean(torch.square(f8))

        if self.iter % 100 == 0:
          print("mse_f0=",mse_f0,"mse_f1=",mse_f1,"mse_f2=",mse_f2,"mse_f3=",mse_f3,"mse_f4=",mse_f4,"mse_f5=",mse_f5,"mse_f6=",mse_f6,"mse_f7=",mse_f7,"mse_f8=",mse_f8)

        mse_pde = mse_f0 + mse_f1 + mse_f2 + mse_f3 + mse_f4 + mse_f5+mse_f6 + mse_f7 + mse_f8

        return mse_pde
    
    def closure(self):

        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        mse_bc = self.bc_loss(xy_bnd)
        mse_outlet = self.outlet_loss(xy_outlet)
        mse_pde = self.pde_loss(xy_col)

        loss =100* mse_bc + mse_outlet + mse_pde

        loss.backward()

        self.losses["bc"].append(mse_bc.detach().cpu().item())
        self.losses["outlet"].append(mse_outlet.detach().cpu().item())
        self.losses["pde"].append(mse_pde.detach().cpu().item())

        self.iter += 1

        print(
            f"\r It: {self.iter} Loss: {loss.item():.5e} BC: {mse_bc.item():.3e} outlet: {mse_outlet.item():.3e} pde: {mse_pde.item():.3e}",
            end="",
        )

        if self.iter % 100 == 0:
            print("")

        return loss

if __name__ == "__main__":

    pinn = PINN()
    start_time = time.time()

    for i in range(10000):
        pinn.closure()
        pinn.adam.step()
    pinn.lbfgs.step(pinn.closure)

    print("--- %s seconds ---" % (time.time() - start_time))
    print(f'-- {(time.time() - start_time)/60} mins --')
    torch.save(pinn.net.state_dict(), "/content/Param.pt")
    plotLoss(pinn.losses, "/content/LossCurve.png", ["BC", "Outlet", "PDE"])

pinn = PINN()

pinn.net.load_state_dict(torch.load("/content/Param.pt"))

x = np.arange(x_min, x_max, 0.001)
y = np.arange(y_min, y_max, 0.001)

X, Y = np.meshgrid(x, y)

x = X.reshape(-1, 1)
y = Y.reshape(-1, 1)

dst_from_cyl = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
cyl_mask = dst_from_cyl > r

xy = np.concatenate([x, y], axis=1)
xy = torch.tensor(xy, dtype=torch.float32).to(device)

with torch.no_grad():
    u, v,T, p, sig_xx, sig_xy, sig_yy,q_x,q_y = pinn.predict(xy)

    u = u.cpu().numpy()
    u = np.where(cyl_mask, u, np.nan).reshape(Y.shape)

    v = v.cpu().numpy()
    v = np.where(cyl_mask, v, np.nan).reshape(Y.shape)

    T = T.cpu().numpy()
    T = np.where(cyl_mask, T, np.nan).reshape(Y.shape)

    p = p.cpu().numpy()
    p = np.where(cyl_mask, p, np.nan).reshape(Y.shape)

fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
data = (u, v,T, p)
labels = ["$u(x,y)$", "$v(x,y)$", "$T(x,y)$", "$p(x,y)$"]
for i in range(4):
    ax = axes[i]
    im = ax.imshow(
        data[i], cmap="rainbow", extent=[x_min, x_max, y_min, y_max], origin="lower"
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad="3%")
    fig.colorbar(im, cax=cax, label=labels[i])
    ax.set_title(labels[i])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")

fig.tight_layout()

fig.savefig("/content/Sol.png", dpi=500)



################################################################################################
################################################################################################
'''

################################################################################################
################################################################################################

'''

