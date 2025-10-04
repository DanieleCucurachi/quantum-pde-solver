# TODO: write proper test script with parser
# TODO: gathe r all parameters related to circuit in one and all parameters related to train in other

import os
import sys
import numpy as np
import pandas as pd

from pathlib import Path

from scipy.optimize import minimize

from src.cost_function import Burgers2D        # your class
from src.ansatz import HEAnsatz # your simple ansatz
from src.utils import set_seeds, fidelity, gaussian_state  # your utility
from src.plot import time_evolution_dataframe_2d, plot_time_evolution_2d

from qiskit.quantum_info import Statevector

def optimize_step(cost_obj, init_params):
    """Optimize cost function for one time step."""
    def obj_fn(lambdas):
        return cost_obj.cost(lambdas)
    res = minimize(obj_fn, init_params, method="COBYLA")  # or SPSA for hardware
    return res.x, res.fun


def main():
    # Essentials (feel free to tweak)
    n_qubits = 4           # grid size N = 2^n
    depth = 2              # ansatz layers
    tau = 0.02             # Euler time step
    nu = 0.1               # kinematic viscosity
    tau = 0.5
    tmax = 1.0
    nsteps = int(tmax/tau)
    domain=[(0.0, 1.0), (0.0, 1.0)]
    
    print("\n ------ Simulation Specs ------ \n")
    # set seed for reproducibility
    set_seeds(seed=42, verbose=True)

    # TODO: I don't think here I need to account for lambda0, at this point does not matter
    # later it will be needed to rescale the state
    # Prepare initial state: Gaussian 
    ansatz = HEAnsatz(n_qubits, depth)
    target = gaussian_state(n_qubits, domain=domain)
    init_params = ansatz.random_params()
    res = minimize(fidelity, init_params, args=(ansatz, target),
               method="COBYLA", options={"maxiter": 200})
    
    lambdas = res.x
    print("\n  ------ Initial State Preparation ------\n")
    print("Optimal λ parameters:", res.x)
    print("\nFinal fidelity:", -res.fun)

    # TODO: with what lambda0 should I initialize?
    # initialize from Gaussian state f(x,y)
    lambda0 = 1.0
    cost_obj = Burgers2D(lambda0, lambdas, nu, tau, n_qubits, depth)

    params = np.concatenate(([lambda0], lambdas))

    # Data logging
    rows = []
    save_every = 1  # adjust if you want fewer writes  # TODO: has to go, save only once at the end
    exp_path = Path("examples/exp_results/burgers")
    exp_path.mkdir(parents=True, exist_ok=True)
    csv_path = exp_path / "data_2d.csv"  # will save in the current working directory

    print("\n  ------ Time Evolution ------\n")
    for step in range(nsteps):
        t = step * tau
        print(f"Step {step}/{nsteps}, time={t:.2f}")

        # One optimization step (your existing function)
        params, cost_val = optimize_step(cost_obj, params)

        # Record a row for this step
        rows.append(
            {
                "step": step,
                "time": t,
                "lambda0": float(params[0]),
                "lambdas": params[1:],
                "cost": float(cost_val),
            }
        )

        # Periodically save a CSV checkpoint
        if (step + 1) % save_every == 0:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)

    print("\n ------ Results Processing ------\n")
    # TODO: do we need this?
    # Final save (ensure last step is persisted)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path.resolve()}")

    # Optional: plot results
    df_funcs = time_evolution_dataframe_2d(df, n_qubits, depth, domain)
    plot_time_evolution_2d(df_funcs, outfile=str(exp_path / "time_evo_2d.png"))
    
    # TODO. in the plots I do not see the last point t=5, why?

if __name__ == "__main__":
    main()