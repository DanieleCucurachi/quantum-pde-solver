from __future__ import annotations


import numpy as np
from abc import ABC, abstractmethod

from fipy import (
    CellVariable,
    Grid1D,
    Grid2D,
    TransientTerm,
    DiffusionTerm,
    ConvectionTerm,
    FaceVariable,
)


class BaseFiPySolver(ABC):
    """
    Base class for PDE solvers using FiPy.
    Handles mesh/domain setup and Gaussian initial condition.
    """

    def __init__(
        self,
        dim: int = 1, 
        nx: int = 10, 
        ny: int = None, 
        Lx: float = 1.0, 
        Ly: float = 1.0,
    ) -> None:
        """
        Initialize solver domain and mesh.

        Args:
            dim (int): Dimension (1 or 2).
            nx (int): Number of grid points in x.
            ny (int): Number of grid points in y (only for 2D, defaults to nx).
            Lx (float): Length of domain in x.
            Ly (float): Length of domain in y (only for 2D).
        """
        self.dim = dim
        self.Lx, self.Ly = Lx, Ly
        self.nx = nx
        self.ny = ny if ny is not None else nx

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

    def gaussian_init(
        self, 
        sigma: float = 0.15, 
        center: tuple[float, ...] = None,
    ) -> np.ndarray:
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

    @abstractmethod
    def solve(self, *args, **kwargs):
        """
        Abstract solve method to be implemented by subclasses.
        """
        pass


class BurgersFiPySolver(BaseFiPySolver):
    """
    FiPy solver for the Burgers' equation in 1D or 2D.
    """

    def solve(
        self,
        nu: float = 0.1,
        dt: float = 0.01,
        steps: int = 100,
        sigma: float = 0.15,
        center: tuple[float, ...] = None,
    ) -> list[np.ndarray]:
        """
        Solve Burgers' equation.

        Args:
            nu (float): Viscosity.
            dt (float): Time step.
            steps (int): Number of time steps.
            sigma (float): Gaussian width for initial condition.
            center (tuple): Gaussian center.

        Returns:
            history (list[np.ndarray]): List of solutions at each step.
        """
        u0 = self.gaussian_init(sigma=sigma, center=center)
        u = CellVariable(name="u", mesh=self.mesh, value=u0, hasOld=True)

        nfaces = self.mesh.numberOfFaces
        if self.dim == 1:
            vel = FaceVariable(mesh=self.mesh, rank=1, value=(np.zeros(nfaces),))
        else:
            vel = FaceVariable(mesh=self.mesh, rank=1, value=(np.zeros(nfaces), np.zeros(nfaces)))

        eq = TransientTerm() + ConvectionTerm(coeff=vel) - DiffusionTerm(coeff=nu)

        history = [u.value.copy()]
        u.updateOld()

        for _ in range(steps):
            # vel = 0.5 * u_face(previous)
            uf = u.old.faceValue
            if self.dim == 1:
                vel.setValue((0.5 * uf,))
            else:
                vel.setValue((0.5 * uf, 0.5 * uf))

            eq.solve(var=u, dt=dt)
            history.append(u.value.copy())
            u.updateOld()

        return history

class DiffusionFiPySolver(BaseFiPySolver):
    """
    FiPy solver for the diffusion equation in 1D or 2D.
    """

    def solve(
        self,
        D: float = 0.1,
        dt: float = 0.01,
        steps: int = 100,
        sigma: float = 0.15,
        center: tuple[float, ...] = None,
    ) -> list[np.ndarray]:
        """
        Solve diffusion equation.

        Args:
            D (float): Diffusion coefficient.
            dt (float): Time step.
            steps (int): Number of time steps.
            sigma (float): Gaussian width for initial condition.
            center (tuple): Gaussian center.

        Returns:
            history (list[np.ndarray]): List of solutions at each step.
        """
        u0 = self.gaussian_init(sigma=sigma, center=center)
        u = CellVariable(name="u", mesh=self.mesh, value=u0, hasOld=True)

        eq = TransientTerm() - DiffusionTerm(coeff=D)

        history = [u.value.copy()]
        u.updateOld()

        for _ in range(steps):
            eq.solve(var=u, dt=dt)
            history.append(u.value.copy())
            u.updateOld()

        return history






class FiPySolver:
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
        equation="burgers",
        nu=0.1,
        D=0.1,
        dt=0.01,
        steps=100,
        sigma=0.15,
        center=None,
    ):
        """
        Solve PDE.

        Args:
            equation (str): "burgers" or "diffusion".
            nu (float): Viscosity for Burgers.
            D (float): Diffusion coefficient.
            dt (float): Time step.
            steps (int): Number of time steps.
            sigma (float): Gaussian width for initial condition.
            center (tuple): Gaussian center.

        Returns:
            history (list[np.ndarray]): List of solutions at each step.
        """
        u0 = self.gaussian_init(sigma=sigma, center=center)

        # IMPORTANT: enable old values for time-stepping
        u = CellVariable(name="u", mesh=self.mesh, value=u0, hasOld=True)

        if equation == "burgers":
            # Face-centered velocity for ConvectionTerm
            nfaces = self.mesh.numberOfFaces
            if self.dim == 1:
                vel = FaceVariable(mesh=self.mesh, rank=1, value=(np.zeros(nfaces),))
            else:
                vel = FaceVariable(mesh=self.mesh, rank=1,
                                value=(np.zeros(nfaces), np.zeros(nfaces)))

            # Semi-implicit in u, explicit in coeff (vel from previous iterate)
            eq = TransientTerm() + ConvectionTerm(coeff=vel) - DiffusionTerm(coeff=nu)

        elif equation == "diffusion":
            # unchanged diffusion branch
            eq = TransientTerm() - DiffusionTerm(coeff=D)
        else:
            raise ValueError("Unknown equation. Use 'burgers' or 'diffusion'.")

        history = [u.value.copy()]

        # Initialize old = current before the first step
        u.updateOld()

        for _ in range(steps):
            if equation == "burgers":
                # vel = 0.5 * u_face(previous)
                uf = u.old.faceValue
                if self.dim == 1:
                    vel.setValue((0.5 * uf,))
                else:
                    vel.setValue((0.5 * uf, 0.5 * uf))

            eq.solve(var=u, dt=dt)
            history.append(u.value.copy())

            # Prepare for next iteration (needed by Burgers)
            u.updateOld()

        return history