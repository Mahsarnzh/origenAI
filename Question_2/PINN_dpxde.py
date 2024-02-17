"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np


Re = 100
nu = 1 / Re
l = 1 / (2 * nu) - np.sqrt(1 / (4 * nu**2) + 4 * np.pi**2)


def pde(x, u):
    u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:]
    u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
    u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

    v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
    v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
    v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

    p_x = dde.grad.jacobian(u, x, i=2, j=0)
    p_y = dde.grad.jacobian(u, x, i=2, j=1)

    momentum_x = (
        u_vel * u_vel_x + v_vel * u_vel_y + p_x - 1 / Re * (u_vel_xx + u_vel_yy)
    )
    momentum_y = (
        u_vel * v_vel_x + v_vel * v_vel_y + p_y - 1 / Re * (v_vel_xx + v_vel_yy)
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

spatial_domain = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])

# Define boundary conditions
bc_left = dde.icbc.NeumannBC(spatial_domain, lambda x: 0.0, boundary_left)
bc_right = dde.icbc.NeumannBC(spatial_domain, lambda x: 0.0, boundary_right)
bc_bottom = dde.icbc.NeumannBC(spatial_domain, lambda x: 0.0, boundary_bottom)
# boundary_condition_u = dde.icbc.DirichletBC(
#     spatial_domain, lambda x: 1, on_boundary: on_boundary, component=0
# )

# boundary_condition_u = dde.icbc.DirichletBC(
#     spatial_domain, lambda x: 1.0, lambda _, on_boundary: on_boundary, component=0
# )
#

boundary_condition_u = dde.icbc.DirichletBC(
    spatial_domain,
    lambda x: 1.0 if any(dde.utils.isclose(x[0], 0)) else 1.0,
    lambda _, on_boundary: on_boundary,
    component=0
)


boundary_condition_v = dde.icbc.DirichletBC(
    spatial_domain,
    lambda x: 0.0 if any(dde.utils.isclose(x[1], 0)) else 0.0,
    lambda _, on_boundary: on_boundary,
    component=1
)


data = dde.data.PDE(
    spatial_domain,
    pde,
    [bc_left, bc_right, bc_bottom, boundary_condition_u, boundary_condition_v],
    num_domain=500,  # Adjust the number of points in the domain as needed
    num_boundary=400,
    num_test=1024,
)

# Solve the problem

net = dde.nn.FNN([2] + 8 * [50] + [3], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(iterations=30000)
model.compile("L-BFGS")
losshistory, train_state = model.train()



###### Data.npy
file_path = 'Data.npy'
data_array = np.load(file_path, allow_pickle=True)
pressure = data_array.tolist()['pressure']
u_velocity = data_array.tolist()['u_velocity']
v_velocity = data_array.tolist()['v_velocity']
x = data_array.tolist()['x']
y = data_array.tolist()['y']
P_star = pressure  # N x T
######


# Rearrange Data
XX = x  # N x T
YY = y  # N x T
# TT = np.tile(t_test[0:20], (1, N)).T  # N x T

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


f = model.predict(X, operator=pde)

l2_difference_u = dde.metrics.l2_relative_error(u_exact, u_pred)
l2_difference_v = dde.metrics.l2_relative_error(v_exact, v_pred)
l2_difference_p = dde.metrics.l2_relative_error(p_exact, p_pred)
residual = np.mean(np.absolute(f))

print("Mean residual:", residual)
