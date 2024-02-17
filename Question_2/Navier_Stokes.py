import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

class NavierStokesSimulation:
    def __init__(self, n_points=50, domain_size=1.0, n_iterations=500, time_step_length=0.001,
                 kinematic_viscosity=0.1, density=1.0, horizontal_velocity_top=1.0,
                 n_pressure_poisson_iterations=50, stability_safety_factor=0.5):
        self.n_points = n_points
        self.domain_size = domain_size
        self.n_iterations = n_iterations
        self.time_step_length = time_step_length
        self.kinematic_viscosity = kinematic_viscosity
        self.density = density
        self.horizontal_velocity_top = horizontal_velocity_top
        self.n_pressure_poisson_iterations = n_pressure_poisson_iterations
        self.stability_safety_factor = stability_safety_factor

        self.element_length = self.domain_size / (self.n_points - 1)
        self.x = np.linspace(0.0, self.domain_size, self.n_points)
        self.y = np.linspace(0.0, self.domain_size, self.n_points)

        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.u_prev = np.zeros_like(self.X)
        self.v_prev = np.zeros_like(self.X)
        self.p_prev = np.zeros_like(self.X)

    def central_difference_x(self, f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, 0:-2]) / (2 * self.element_length)
        return diff

    def central_difference_y(self, f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (f[2:, 1:-1] - f[0:-2, 1:-1]) / (2 * self.element_length)
        return diff

    def laplace(self, f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (f[1:-1, 0:-2] + f[0:-2, 1:-1] - 4 * f[1:-1, 1:-1]
                           + f[1:-1, 2:] + f[2:, 1:-1]) / (self.element_length**2)
        return diff

    def run_simulation(self):
        max_possible_time_step_length = 0.5 * self.element_length**2 / self.kinematic_viscosity
        if self.time_step_length > self.stability_safety_factor * max_possible_time_step_length:
            raise RuntimeError("Stability is not guaranteed")

        for _ in tqdm(range(self.n_iterations)):
            d_u_prev__d_x = self.central_difference_x(self.u_prev)
            d_u_prev__d_y = self.central_difference_y(self.u_prev)
            d_v_prev__d_x = self.central_difference_x(self.v_prev)
            d_v_prev__d_y = self.central_difference_y(self.v_prev)
            laplace__u_prev = self.laplace(self.u_prev)
            laplace__v_prev = self.laplace(self.v_prev)

            u_tent = (self.u_prev + self.time_step_length * (
                - (self.u_prev * d_u_prev__d_x + self.v_prev * d_u_prev__d_y)
                + self.kinematic_viscosity * laplace__u_prev
            ))
            v_tent = (self.v_prev + self.time_step_length * (
                - (self.u_prev * d_v_prev__d_x + self.v_prev * d_v_prev__d_y)
                + self.kinematic_viscosity * laplace__v_prev
            ))

            # Velocity Boundary Conditions
            self.apply_velocity_boundary_conditions(u_tent, v_tent)

            d_u_tent__d_x = self.central_difference_x(u_tent)
            d_v_tent__d_y = self.central_difference_y(v_tent)

            rhs = (self.density / self.time_step_length * (d_u_tent__d_x + d_v_tent__d_y))

            p_next = self.solve_pressure_poisson(rhs)
            self.apply_pressure_boundary_conditions(p_next)

            d_p_next__d_x = self.central_difference_x(p_next)
            d_p_next__d_y = self.central_difference_y(p_next)

            u_next = (u_tent - self.time_step_length / self.density * d_p_next__d_x)
            v_next = (v_tent - self.time_step_length / self.density * d_p_next__d_y)

            # Velocity Boundary Conditions
            self.apply_velocity_boundary_conditions(u_next, v_next)

            self.u_prev, self.v_prev, self.p_prev = u_next, v_next, p_next

        self.visualize_results()

    def apply_velocity_boundary_conditions(self, u, v):
        u[0, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        u[-1, :] = self.horizontal_velocity_top
        v[0, :] = 0.0
        v[:, 0] = 0.0
        v[:, -1] = 0.0
        v[-1, :] = 0.0

    def solve_pressure_poisson(self, rhs):
        p_next = np.zeros_like(self.p_prev)
        p_next[1:-1, 1:-1] = 1/4 * (
            p_next[1:-1, 0:-2] + p_next[0:-2, 1:-1] + p_next[1:-1, 2:] + p_next[2:, 1:-1]
            - self.element_length**2 * rhs[1:-1, 1:-1]
        )
        # Pressure Boundary Conditions
        p_next[:, -1] = p_next[:, -2]
        p_next[0, :] = p_next[1, :]
        p_next[:, 0] = p_next[:, 1]
        p_next[-1, :] = 0.0
        return p_next

    def apply_pressure_boundary_conditions(self, p):
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[:, 0] = p[:, 1]
        p[-1, :] = 0.0

    def visualize_results(self):
        plt.figure()
        sns.heatmap(self.p_prev[::2, ::2], cmap="viridis", cbar=True)

        # Quiver plot for vectors
        plt.quiver(self.X[::2, ::2], self.Y[::2, ::2], self.u_prev[::2, ::2], self.v_prev[::2, ::2], color="red")

        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.show()



def main():
    simulation = NavierStokesSimulation()
    simulation.run_simulation()

if __name__ == "__main__":
    main()
