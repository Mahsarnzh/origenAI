"""
@author: Mahsa Raeisinezhad
"""
import sys

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

Re = 100
nu = 1 / Re


# noinspection PyPep8Naming
class NavierStokes:
    def __init__(self, X, Y, u, v):

        self.x = (
            torch.tensor(X, dtype=torch.float32).clone().detach().requires_grad_(True)
        )
        self.y = (
            torch.tensor(Y, dtype=torch.float32).clone().detach().requires_grad_(True)
        )

        self.u = torch.tensor(u, dtype=torch.float32)
        self.v = torch.tensor(v, dtype=torch.float32)

        # null vector to test against f and g:
        self.null = torch.zeros((self.x.shape[0], 1))

        # initialize network:
        self.network()

        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1,
            max_iter=2000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-05,
            tolerance_change=0.5 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )

        self.mse = nn.MSELoss()

        # loss
        self.ls = 0

        # iteration number
        self.iter = 0

    def network(self):

        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 2),
        )

    def forward(self, x, y):

        res = self.net(torch.hstack((x, y)))
        psi, p = res[:, 0:1], res[:, 1:2]

        u = torch.autograd.grad(
            psi, y, grad_outputs=torch.ones_like(psi), create_graph=True
        )[
            0
        ]  # retain_graph=True,
        v = (
            -1.0
            * torch.autograd.grad(
                psi, x, grad_outputs=torch.ones_like(psi), create_graph=True
            )[0]
        )

        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, y, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True
        )[0]

        v_x = torch.autograd.grad(
            v, x, grad_outputs=torch.ones_like(v), create_graph=True
        )[0]
        v_xx = torch.autograd.grad(
            v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True
        )[0]
        v_y = torch.autograd.grad(
            v, y, grad_outputs=torch.ones_like(v), create_graph=True
        )[0]
        v_yy = torch.autograd.grad(
            v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True
        )[0]

        p_x = torch.autograd.grad(
            p, x, grad_outputs=torch.ones_like(p), create_graph=True
        )[0]
        p_y = torch.autograd.grad(
            p, y, grad_outputs=torch.ones_like(p), create_graph=True
        )[0]

        momentum_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        momentum_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        continuity = u_x + v_y

        return u, v, p, momentum_x, momentum_y, continuity

    def closure(self):
        # reset gradients to zero:
        self.optimizer.zero_grad()

        # u, v, p, g and f predictions:
        (
            u_prediction,
            v_prediction,
            p_prediction,
            momentum_x_pred,
            momentum_y_pred,
            continuity_pred,
        ) = self.forward(self.x, self.y)

        # calculate losses
        u_loss = self.mse(u_prediction, self.u)
        v_loss = self.mse(v_prediction, self.v)
        f_loss = self.mse(momentum_x_pred, self.null)
        g_loss = self.mse(momentum_y_pred, self.null)
        continuity_loss = self.mse(continuity_pred, self.null)
        self.ls = u_loss + v_loss + f_loss + g_loss + continuity_loss

        # derivative with respect to net's weights:
        self.ls.backward()

        self.iter += 1
        if not self.iter % 1:
            print("Iteration: {:}, Loss: {:0.6f}".format(self.iter, self.ls))

        return self.ls

    def train(self, num_epochs=100):
        # Training loop
        self.net.train()

        for epoch in range(num_epochs):
            self.optimizer.step(self.closure)
            print("Epoch: {:}, Loss: {:0.6f}".format(epoch + 1, self.ls))

    def plot_results(self, x_test, y_test, x, y, u_velocity, v_velocity, pressure):
        self.net.eval()

        u_out, v_out, p_out, _, _, _ = self.forward(x_test, y_test)

        sns.set(style="whitegrid")  # Set background style
        plt.figure()
        sns.heatmap(pressure, cmap="viridis", cbar=True)
        plt.quiver(x, y, u_velocity, v_velocity, color="red", scale=20)
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.show()

        # Plot pressure using heatmap
        plt.figure(figsize=(8, 6))
        p_out_np = p_out.detach().numpy()
        sns.heatmap(p_out_np, cmap="viridis", cbar=True)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Pressure Field")

        plt.show()


def main():
    # Create and train the model
    N_train = 5000

    # Data.npy
    file_path = "Data.npy"
    data_array = np.load(file_path, allow_pickle=True)
    pressure = data_array.tolist()["pressure"]
    u_velocity = data_array.tolist()["u_velocity"]
    v_velocity = data_array.tolist()["v_velocity"]
    x = data_array.tolist()["x"]
    y = data_array.tolist()["y"]
    P_star = pressure  # N x N


    N = x.shape[0]

    x_test = x[:, 0:1]
    y_test = y[:, 0:1]
    x_test = x_test.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    p_test = pressure[:, 0]
    u_test = u_velocity[:, 0]
    # t_test = np.ones((x_test.shape[0], x_test.shape[1]))

    # Rearrange Data
    XX = x  # N x N
    YY = y  # N x N

    UU = u_velocity  # N x N
    VV = v_velocity  # N x N
    PP = pressure  # N x N

    x = XX.flatten()[:, None]  # NN x 1
    y = YY.flatten()[:, None]  # NN x 1
    # t = TT.flatten()[:, None]  # NN x 1

    u = UU.flatten()[:, None]  # NN x 1
    v = VV.flatten()[:, None]  # NN x 1
    p = PP.flatten()[:, None]  # NN x 1

    # Training Data
    idx = np.random.choice(10, 100, replace=True)
    x_train = x[idx, :]
    y_train = y[idx, :]
    u_train = u[idx, :]
    v_train = v[idx, :]

    pinn = NavierStokes(x_train, y_train, u_train, v_train)
    pinn.train()
    torch.save(pinn.net.state_dict(), "model.pt")

    # pinn.net.load_state_dict(torch.load('model.pt'))
    pinn.net.eval()

    x_test = torch.tensor(x_test, dtype=torch.float32, requires_grad=True)
    y_test = torch.tensor(y_test, dtype=torch.float32, requires_grad=True)

    # x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True)
    # y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True)

    pinn.train(num_epochs=100)

    # Save the trained model
    torch.save(pinn.net.state_dict(), "model.pt")
    pinn.net.eval()

    # Plot results
    pinn.plot_results(x_test, y_test, x, y, u_velocity, v_velocity, pressure)


if __name__ == "__main__":
    sys.exit(main())
