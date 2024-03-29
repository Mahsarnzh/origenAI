"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""

import matplotlib.pyplot as plt
import deepxde as dde
import numpy as np


Re = 100
nu = 1 / Re
l = 1 / (2 * nu) - np.sqrt(1 / (4 * nu**2) + 4 * np.pi**2)

def pde(x, u):
    u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:]
    u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)  # i --> x component
    u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)  # j --> y component
    u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

    v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
    v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
    v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

    p_x = dde.grad.jacobian(u, x, i=2, j=0)
    p_y = dde.grad.jacobian(u, x, i=2, j=1)

    momentum_x = (
        u_vel * u_vel_x + v_vel * u_vel_y + p_x - nu * (u_vel_xx + u_vel_yy)
    )
    momentum_y = (
        u_vel * v_vel_x + v_vel * v_vel_y + p_y - nu * (v_vel_xx + v_vel_yy)
    )
    continuity = u_vel_x + v_vel_y

    return [momentum_x, momentum_y, continuity]


def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)


def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)


def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0)


def boundary_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 1)


# xmin – Coordinate of bottom left corner.
# xmax – Coordinate of top right corner.

# Explicitly set orientation to ensure consistent color mapping
spatial_domain = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])
# Define boundary conditions
bc_left = dde.DirichletBC(spatial_domain, lambda x: [0.0, 0.0, 0.0], boundary_left)
bc_right = dde.DirichletBC(spatial_domain, lambda x: [0.0, 0.0, 0.0], boundary_right)
bc_bottom = dde.DirichletBC(spatial_domain, lambda x: [0.0, 0.0, 0.0], boundary_bottom)

# Homogeneous Dirichlet Boundary condition at the top side
bc_top = dde.DirichletBC(spatial_domain, lambda x: [1.0, 0.0, 0.0], boundary_top)

# Modify the boundary_condition_u and boundary_condition_v functions
boundary_condition_u = dde.DirichletBC(
    spatial_domain,
    lambda x: 1.0 if any(dde.utils.isclose(x[0], 1)) else 0.0,
    lambda x, on_boundary: on_boundary and not dde.utils.isclose(x[1], 1),
    component=0,
)

"""
where x[0]:x is close to 1 and x[1]:y is not close to 1)
It checks if any value in the second coordinate (x[0]:x) is close to 1. If true, it returns 1.0, otherwise, it returns 0.0.

boundary_condition_u sets a modified Dirichlet boundary condition for the u component. 
If any point lies on the top boundary (x[0] close to 1) and v component is not on the top boundary (not dde.utils.isclose(x[1], 1)), 
then the value is set to 1.0; otherwise, it is set to 0.0.
"""


boundary_condition_v = dde.DirichletBC(
    spatial_domain,
    lambda x: 1.0 if any(dde.utils.isclose(x[1], 0)) else 0.0,
    lambda x, on_boundary: on_boundary and not dde.utils.isclose(x[1], 1),
    component=1,
)

"""
It checks if any value in the second coordinate (x[1]:y) is close to 0. If true, it returns 1.0, otherwise, it returns 0.0.
Dirichlet boundary condition for the second variable (component=1) in a PDE problem. 
The condition sets the variable to 1.0 when x[1] is close to 0 on the specified part of the boundary,
 and it is applied only on the specified part of the boundary where x[1] is not close to 1.
"""

data = dde.data.PDE(
    spatial_domain,
    pde,
    [bc_left, bc_right, bc_bottom, bc_top, boundary_condition_u, boundary_condition_v],
    num_domain=500,  # Adjust the number of points in the domain as needed
    num_boundary=400,
    num_test=1024,
)

# Solve the problem
net = dde.nn.FNN([2] + 8 * [50] + [3], "tanh", "Glorot normal")

model = dde.Model(data, net)
model.compile("L-BFGS")
losshistory, train_state = model.train(iterations=40)

###### Data.npy
file_path = "Data.npy"
data_array = np.load(file_path, allow_pickle=True)
pressure = data_array.tolist()["pressure"]
u_velocity = data_array.tolist()["u_velocity"]
v_velocity = data_array.tolist()["v_velocity"]
x = data_array.tolist()["x"]
y = data_array.tolist()["y"]
P_star = pressure  # N x N
######

# Rearrange Data
XX = x  # N x N
YY = y  # N x N

UU = u_velocity  # N x N
VV = v_velocity  # N x N
PP = pressure  # N x N

x = XX.flatten()[:, None]  # NN x 1
y = YY.flatten()[:, None]  # NN x 1
X = np.hstack((x, y))
u = UU.flatten()[:, None]  # NT x 1
v = VV.flatten()[:, None]  # NT x 1
p = PP.flatten()[:, None]  # NT x 1

output = model.predict(X)
u_pred = output[:, 0]
v_pred = output[:, 1]
p_pred = output[:, 2]

u_exact = u.reshape(-1)
v_exact = v.reshape(-1)
p_exact = p.reshape(-1)

# Plot Contours of Prediction vs Data
plt.figure(figsize=(5, 5))

# Plot Exact U Component
plt.subplot(1, 6, 1)
plt.tricontourf(x.flatten(), y.flatten(), u_exact, cmap="viridis", levels=20)
plt.title("Exact U Component")

# Plot Predicted U Component
plt.subplot(1, 6, 2)
plt.tricontourf(x.flatten(), y.flatten(), u_pred, cmap="viridis", levels=20)
plt.title("Predicted U Component")

# Plot Exact V Component
plt.subplot(1, 6, 3)
plt.tricontourf(x.flatten(), y.flatten(), -v_exact, cmap="viridis", levels=20)
plt.title("Exact V Component")

# Plot Predicted V Component
plt.subplot(1, 6, 4)
plt.tricontourf(x.flatten(), y.flatten(), v_pred, cmap="viridis", levels=20)
plt.title("Predicted V Component")

# Plot Exact Pressure
plt.subplot(1, 6, 5)
plt.tricontourf(x.flatten(), y.flatten(), p_exact, cmap="viridis", levels=20)
plt.title("Exact Pressure")

# Plot Predicted Pressure
plt.subplot(1, 6, 6)
plt.tricontourf(x.flatten(), y.flatten(), p_pred, cmap="viridis", levels=20)
plt.title("Predicted Pressure")

plt.show()


plt.figure(figsize=(10, 5))
plt.plot(losshistory.steps, losshistory.loss_train, label="Train Loss")
plt.plot(losshistory.steps, losshistory.loss_test, label="Test Loss")
plt.title("Loss History")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
