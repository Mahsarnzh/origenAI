"""
@author: Mahsa Raeisinezhad
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

Re = 100
nu = 1 / Re


class NavierStokes:
    def __init__(self, X, Y, u, v, p):
        self.x = (
            torch.tensor(X, dtype=torch.float32).clone().detach().requires_grad_(True)
        )
        self.y = (
            torch.tensor(Y, dtype=torch.float32).clone().detach().requires_grad_(True)
        )

        self.loss_history = []
        self.u = torch.tensor(u, dtype=torch.float32)
        self.v = torch.tensor(v, dtype=torch.float32)
        self.p = torch.tensor(p, dtype=torch.float32)
        # null vector to test against f and g:
        self.null = torch.zeros((self.x.shape[0], 1))

        # initialize network:
        self.net = self.get_network()

        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1,
            max_iter=200,
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

    @staticmethod
    def get_network():
        return nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 2),
            nn.Tanh(),  # Add an activation function to squash values between -1 and 1
        )

    def function(self, x, y):
        res = self.net(torch.hstack((x, y)))
        psi, p = res[:, 0:1], res[:, 1:2]

        u = torch.autograd.grad(
            psi, y, grad_outputs=torch.ones_like(psi), create_graph=True
        )[
            0
        ]
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
        ) = self.function(self.x, self.y)

        # calculate losses with different weights
        u_loss = self.mse(u_prediction, self.u)
        v_loss = self.mse(v_prediction, self.v)
        p_loss = self.mse(p_prediction, self.p)
        f_loss = self.mse(momentum_x_pred, self.null)
        g_loss = self.mse(momentum_y_pred, self.null)
        continuity_loss = self.mse(continuity_pred, self.null)

        # Assign different weights to losses
        total_loss = (
                10 * u_loss +
                10 * v_loss +
                10 * p_loss +
                100 * f_loss +
                100 * g_loss +
                100 * continuity_loss
        )

        # derivative with respect to net's weights:
        total_loss.backward()

        self.iter += 1
        if not self.iter % 1:
            print("Iteration: {:}, Loss: {:0.6f}".format(self.iter, total_loss))
            self.loss_history.append(total_loss.item())

        return total_loss

    def train(self, num_epochs=1000):
        # Training loop
        self.net.train()

        for epoch in range(num_epochs):
            self.optimizer.step(self.closure)
            print("Epoch: {:}, Loss: {:0.6f}".format(epoch + 1, self.ls))

    @staticmethod
    def plot_contours(u_out_np, u, p_out_np, p, v_out_np, v, XX, YY, loss_history, x_test, y_test):
        # Plot Contours of Prediction vs Data
        plt.figure(figsize=(15, 5))

        # Plot Exact U Component
        plt.subplot(1, 6, 1)
        side_length = int(np.sqrt(u.shape[0]))
        u_reshaped = u.reshape(side_length, side_length)
        plt.contourf(u_reshaped, cmap=cm.viridis)
        plt.tricontourf(XX.flatten(), YY.flatten(), u_reshaped.flatten(), cmap="viridis", levels=20)
        plt.title("Exact U Component")

        # Plot pred U Component
        plt.subplot(1, 6, 2)
        side_length = int(np.sqrt(u_out_np.shape[0]))
        u_out_np_reshaped = u_out_np.reshape(side_length, side_length)
        plt.contourf(u_out_np_reshaped, cmap=cm.viridis)
        plt.tricontourf(x_test.detach().numpy().flatten(), y_test.detach().numpy().flatten(),
                        u_out_np_reshaped.flatten(), cmap="viridis", levels=20)
        plt.title("Predicted U Component")

        # Plot Exact V Component
        plt.subplot(1, 6, 3)
        side_length = int(np.sqrt(v.shape[0]))
        v_reshaped = v.reshape(side_length, side_length)
        plt.contourf(v_reshaped, cmap=cm.viridis)
        plt.tricontourf(XX.flatten(), YY.flatten(), v_reshaped.flatten(), cmap="viridis", levels=20)
        plt.title("Exact V Component")

        # Plot Predicted V Component
        plt.subplot(1, 6, 4)
        side_length = int(np.sqrt(v_out_np.shape[0]))
        v_out_np_reshaped = v_out_np.reshape(side_length, side_length)
        plt.contourf(v_out_np_reshaped, cmap=cm.viridis)
        plt.tricontourf(x_test.detach().numpy().flatten(), y_test.detach().numpy().flatten(), v_out_np_reshaped.flatten(), cmap="viridis", levels=20)
        plt.title("Predicted V Component")

        # Plot Exact P Component
        plt.subplot(1, 6, 5)
        side_length = int(np.sqrt(p.shape[0]))
        p_np_reshaped = p.reshape(side_length, side_length)
        plt.contourf(p_np_reshaped, cmap=cm.viridis)
        plt.tricontourf(XX.flatten(), YY.flatten(), p_np_reshaped.flatten(), cmap="viridis", levels=20)
        plt.title("Exact Pressure")

        # Plot Predicted P Component
        plt.subplot(1, 6, 6)
        side_length = int(np.sqrt(p_out_np.shape[0]))
        p_out_np_reshaped = p_out_np.reshape(side_length, side_length)
        plt.contourf(p_out_np_reshaped, cmap=cm.viridis)
        plt.tricontourf(x_test.detach().numpy().flatten(), y_test.detach().numpy().flatten(), p_out_np_reshaped.flatten(), cmap="viridis", levels=20)
        plt.title("Predicted Pressure")

        plt.figure()
        plt.plot(loss_history, label="Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Loss History")

        plt.show()


def main():
    # Data.npy
    file_path = "Data.npy"
    data_array = np.load(file_path, allow_pickle=True)
    pressure = data_array.tolist()["pressure"]
    u_velocity = data_array.tolist()["u_velocity"]
    v_velocity = data_array.tolist()["v_velocity"]
    x = data_array.tolist()["x"]
    y = data_array.tolist()["y"]
    x_test = x[20:61, 20:61].reshape(-1, 1)
    y_test = y[20:61, 20:61].reshape(-1, 1)

    # Rearrange Data
    XX = x  # N x N
    YY = y  # N x N

    UU = u_velocity  # N x N
    VV = v_velocity  # N x N

    x = XX.flatten()[:, None]  # NN x 1
    y = YY.flatten()[:, None]  # NN x 1

    u = UU.flatten()[:, None]  # NN x 1
    v = VV.flatten()[:, None]  # NN x 1
    p = pressure.flatten()[:, None]

    # Training Data, I changed to 1000 so that the model predicts more accurate for contour plots
    num_data_points = 100
    idx = np.random.choice(3721, num_data_points, replace=False)
    x_train = x[idx, :]
    y_train = y[idx, :]
    u_train = u[idx, :]
    v_train = v[idx, :]
    p_train = p[idx, :]
    # x_test = x_test.reshape(-1, 1)
    # y_test = y_test.reshape(-1, 1)
    x_test = x.reshape(-1, 1)
    y_test = y.reshape(-1, 1)
    pinn = NavierStokes(x_train, y_train, u_train, v_train, p_train)
    pinn.net.eval()

    x_test = torch.tensor(x_test, dtype=torch.float32, requires_grad=True)
    y_test = torch.tensor(y_test, dtype=torch.float32, requires_grad=True)

    # Create and train the model
    pinn.train(num_epochs=100)

    u_out, v_out, p_out, _, _, _ = pinn.function(x_test, y_test)

    # Plot results
    p_out_np = p_out.detach().numpy()
    v_out_np = v_out.detach().numpy()
    u_out_np = u_out.detach().numpy()
    pinn.plot_contours(u_out_np, u, p_out_np, p, v_out_np, v, XX, YY, pinn.loss_history,  x_test, y_test)


if __name__ == "__main__":
    sys.exit(main())