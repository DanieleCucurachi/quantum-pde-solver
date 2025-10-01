from __future__ import annotations

import numpy as np

from fipy import (
    CellVariable,
    Grid1D,
    Grid2D,
    TransientTerm,
    DiffusionTerm,
    ConvectionTerm,
)


# TODO: use better approach to check dimensions is right
class PDESolver:
    """
    PDE Solver for Burgers' and Diffusion equations in 1D and 2D using FiPy.
    """

    def __init__(self, dim=1, nx=10, ny=10, Lx=1.0, Ly=1.0):
        """
        Initialize solver domain and mesh.

        Args:
            dim (int): Dimension (1 or 2).
            nx (int): Number of grid points in x.
            ny (int): Number of grid points in y (only for 2D).
            Lx (float): Length of domain in x.
            Ly (float): Length of domain in y (only for 2D).
        """
        self.dim = dim
        self.Lx, self.Ly = Lx, Ly
        self.nx, self.ny = nx, ny if ny else nx

        if dim == 1:
            dx = Lx / nx
            self.mesh = Grid1D(nx=nx, dx=dx)
            self.x = self.mesh.cellCenters[0]
        elif dim == 2:
            dx = Lx / nx
            dy = Ly / self.ny
            self.mesh = Grid2D(nx=nx, ny=self.ny, dx=dx, dy=dy)
            self.x, self.y = self.mesh.cellCenters
        else:
            raise ValueError("Only 1D or 2D supported.")

    def gaussian_init(self, sigma=0.15, center=None):
        """
        Create Gaussian initial condition.

        Args:
            sigma (float): Std deviation of Gaussian.
            center (tuple): Center of Gaussian (default: domain center).
        """
        if self.dim == 1:
            c = center if center is not None else self.Lx / 2
            u0 = np.exp(-((self.x - c) ** 2) / (2 * sigma**2))
        else:
            cx, cy = center if center is not None else (self.Lx / 2, self.Ly / 2)
            u0 = np.exp(-(((self.x - cx) ** 2 + (self.y - cy) ** 2)) / (2 * sigma**2))

        # Normalize L2 norm
        norm = np.sqrt((u0**2).sum())
        u0 /= norm
        return u0

    # TODO: DELETE, DO NOT WORK
    def solve(
        self,
        equation: str = "burgers",
        nu: float = 0.1,       # viscosity for Burgers
        D: float = 0.1,        # diffusion coefficient
        dt: float = 0.01,
        steps: int = 100,
        sigma: float = 0.15,
        center=None,
    ):
        """
        Solve PDE.

        Args:
            equation: "burgers" or "diffusion".
            nu: Viscosity for Burgers.
            D: Diffusion coefficient.
            dt: Time step.
            steps: Number of time steps.
            sigma: Gaussian width for initial condition.
            center: Gaussian center.

        Returns:
            history: list[np.ndarray], solution snapshots (including t=0).
        """
        # Initial condition
        u0 = self.gaussian_init(sigma=sigma, center=center)
        # IMPORTANT: hasOld=True so we can use u.updateOld() and u.old.faceValue
        u = CellVariable(name="u", mesh=self.mesh, value=u0, hasOld=True)

        history = [u.value.copy()]

        if equation == "burgers":
            # Split step: explicit convection (uses u.old.faceValue), then implicit diffusion.
            for _ in range(steps):
                # Save previous solution for explicit convection
                u.updateOld()

                # 1) Explicit convection update
                # 1D: convection coeff is a single face field; 2D: tuple (vx, vy)
                if self.dim == 1:
                    conv_coeff = (u.old.faceValue,)
                else:
                    conv_coeff = (u.old.faceValue, u.old.faceValue)

                convection_eq = TransientTerm() + ConvectionTerm(coeff=conv_coeff)
                convection_eq.solve(var=u, dt=dt)

                # 2) Implicit diffusion update
                diffusion_eq = TransientTerm() - DiffusionTerm(coeff=nu)
                diffusion_eq.solve(var=u, dt=dt)

                history.append(u.value.copy())

        elif equation == "diffusion":
            # Standard implicit diffusion scheme
            eq = TransientTerm() - DiffusionTerm(coeff=D)
            for _ in range(steps):
                eq.solve(var=u, dt=dt)
                history.append(u.value.copy())

        else:
            raise ValueError("Unknown equation. Use 'burgers' or 'diffusion'.")
        
        return history


# # WORKS FOR DIFFUSION, not for burgers
# def solve(
#         self,
#         equation="burgers",
#         nu=0.1,
#         D=0.1,
#         dt=0.01,
#         steps=100,
#         sigma=0.15,
#         center=None,
#     ):
#         """
#         Solve PDE.

#         Args:
#             equation (str): "burgers" or "diffusion".
#             nu (float): Viscosity for Burgers.
#             D (float): Diffusion coefficient.
#             dt (float): Time step.
#             steps (int): Number of time steps.
#             sigma (float): Gaussian width for initial condition.
#             center (tuple): Gaussian center.

#         Returns:
#             history (list[np.ndarray]): List of solutions at each step.
#         """
#         u0 = self.gaussian_init(sigma=sigma, center=center)
#         u = CellVariable(name="u", mesh=self.mesh, value=u0)

#         # Define PDE
#         if equation == "burgers":
#             if self.dim == 1:
#                 eq = TransientTerm() + ConvectionTerm(coeff=u) - DiffusionTerm(coeff=nu)
#             else:
#                 convection_coeff = (u, u)  # (ux, uy)
#                 eq = TransientTerm() + ConvectionTerm(coeff=convection_coeff) - DiffusionTerm(coeff=nu)
#         elif equation == "diffusion":
#             eq = TransientTerm() - DiffusionTerm(coeff=D)
#         else:
#             raise ValueError("Unknown equation. Use 'burgers' or 'diffusion'.")

#         # Time stepping
#         history = [u.value.copy()]
#         for _ in range(steps):
#             eq.solve(var=u, dt=dt)
#             history.append(u.value.copy())

#         return history