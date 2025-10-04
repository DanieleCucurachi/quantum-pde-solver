from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.ansatz import HEAnsatz
from src.utils import fidelity, gaussian_state
from src.pdes import BasePDE


def prepare_initial_state(
    n_qubits: int,
    depth: int,
    domain: list[tuple[float, float]],
    sigma: float = 0.15
) -> tuple[np.ndarray, float]:
    """
    Optimize the variational ansatz to prepare a normalized Gaussian initial state.

    Args:
        n_qubits (int): Number of qubits (defines grid size).
        depth (int): Number of ansatz layers.
        domain (list[tuple[float, float]]): Domain for the grid (1D or 2D).
        sigma (float, optional): Standard deviation of the Gaussian. Default is 0.15.

    Returns:
        tuple:
            lambdas (np.ndarray): Optimized ansatz parameters.
            final_fidelity (float): Final fidelity achieved (overlap with target state).
    """
    ansatz = HEAnsatz(n_qubits, depth)
    target = gaussian_state(n_qubits, domain=domain, sigma=sigma)
    init_params = ansatz.random_params()
    res = minimize(fidelity, init_params, args=(ansatz, target), 
                   method="COBYLA", options={"maxiter": 200})
    lambdas = res.x
    final_fidelity = -res.fun
    return lambdas, final_fidelity


def optimize_step(
    pde: BasePDE,
    init_params: np.ndarray | list[float]
) -> tuple[np.ndarray, float]:
    """
    Optimize the cost function for a single time step.

    Args:
        pde (BasePDE): PDE object with a .cost(lambdas) method.
        init_params (np.ndarray | list[float]): Initial parameters for optimization.

    Returns:
        tuple:
            params (np.ndarray): Optimized parameters after one step.
            cost_val (float): Final value of the cost function.
    """
    def obj_fn(lambdas):
        return pde.cost(lambdas)
    res = minimize(obj_fn, init_params, method="COBYLA")
    return res.x, res.fun


def run_time_evolution(
    lambda0: float,
    lambdas: np.ndarray | list[float],
    pde: BasePDE,
    nu: float,
    tau: float,
    n_qubits: int,
    depth: int,
    nsteps: int
) -> pd.DataFrame:
    """
    Run the time evolution loop for the variational PDE solver.

    At each step, the cost function is optimized and the parameters are updated.

    Args:
        lambda0 (float): Initial scaling parameter.
        lambdas (np.ndarray | list[float]): Initial ansatz parameters.
        pde (BasePDE): PDE object with a .cost(lambdas) method.
        nu (float): Viscosity or diffusion parameter.
        tau (float): Time step size.
        n_qubits (int): Number of qubits.
        depth (int): Ansatz depth.
        nsteps (int): Number of time steps.

    Returns:
        pd.DataFrame: DataFrame with columns ["step", "time", "lambda0", "lambdas", "cost"] 
                      for each step.
    """
    params = np.concatenate(([lambda0], lambdas))
    rows = []

    # Add initial state before optimization (step 0, time 0)
    rows.append({
        "step": 0,
        "time": 0.0,
        "lambda0": float(params[0]),
        "lambdas": params[1:].copy(),  # store initial lambdas
         "cost": None,  # or compute cost if desired
    })
    print(f"Step 0/{nsteps}, time=0.00")

    for step in range(nsteps):
        t = (step + 1) * tau
        print(f"Step {step + 1}/{nsteps}, time={t:.2f}")
        params, cost_val = optimize_step(
            pde=pde(lambda0, lambdas, nu, tau, n_qubits, depth),
            init_params=params
        )
        rows.append({
            "step": step,
            "time": t,
            "lambda0": float(params[0]),
            "lambdas": params[1:],
            "cost": float(cost_val),
        })
    df = pd.DataFrame(rows)
    return df