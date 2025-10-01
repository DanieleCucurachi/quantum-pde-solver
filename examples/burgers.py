# TODO: write proper test script with parser

#!/usr/bin/env python3
# Minimal Burgers optimization script (SciPy only), using your existing CostFunction.

import os
import sys
import numpy as np
import pandas as pd

from pathlib import Path

from scipy.optimize import minimize

# # Make sure src/ is on path
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
# if SRC_DIR not in sys.path:
#     sys.path.insert(0, SRC_DIR)

from src.cost_function import CostFunction        # your class
from src.variational import SingleParameterAnsatz, Ansatz # your simple ansatz
from src.utils import set_seeds, fidelity, gaussian_state  # your utility

from qiskit.quantum_info import Statevector

def optimize_step(cost_obj, init_params):
    """Optimize cost function for one time step."""
    def obj_fn(lambdas):
        return cost_obj.compute_cost_via_qc(lambdas)
    res = minimize(obj_fn, init_params, method="COBYLA")  # or SPSA for hardware
    return res.x, res.fun


def main():
    # Essentials (feel free to tweak)
    n_qubits = 4           # grid size N = 2^n
    depth = 2              # ansatz layers
    tau = 0.02             # Euler time step
    nu = 0.1               # kinematic viscosity
    tau = 0.5
    tmax = 5.0
    nsteps = int(tmax/tau)
    domain = [0.0, 1.0]
    
    print("\n ------ Simulation Specs ------ \n")
    # set seed for reproducibility
    set_seeds(seed=42, verbose=True)

    # TODO: I don't think here I need to account for lambda0, at this point does not matter
    # later it will be needed to rescale the state
    # Prepare initial state: Gaussian 
    ansatz = SingleParameterAnsatz(n_qubits, depth)
    target = gaussian_state(n_qubits)
    init_param = ansatz.random_param()
    res = minimize(fidelity, init_param, args=(ansatz, target),
               method="COBYLA", options={"maxiter": 200})
    
    lambda1 = res.x[0]
    print("\n  ------ Initial State Preparation ------\n")
    print("Optimal λ parameters:", res.x)
    print("Final fidelity:", -res.fun)

    # TODO: with what lambda0 should I initialize?
    # initialize from Gaussian state f(x,y)
    lambda0 = 1.0
    cost_obj = CostFunction(lambda0, lambda1, nu, tau, n_qubits, depth)

    params = [lambda0, lambda1]

    # Data logging
    rows = []
    save_every = 1  # adjust if you want fewer writes
    exp_path = Path("exp_results/burgers")
    exp_path.mkdir(parents=True, exist_ok=True)
    csv_path = exp_path / "data.csv"  # will save in the current working directory

    print("\n  ------ Time Evolution ------\n")
    for step in range(nsteps):
        t = step * tau
        print(f"Step {step}/{nsteps}, time={t:.2f}")

        # One optimization step (your existing function)
        params, cost_val = optimize_step(cost_obj, params)

        # Unpack parameters explicitly for clarity
        lam0, lam1 = float(params[0]), float(params[1])

        # Record a row for this step
        rows.append(
            {
                "step": step,
                "time": t,
                "lambda0": lam0,
                "lambda": lam1,
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
    from src.plot import time_evolution_dataframe, plot_time_evolution
    df_funcs = time_evolution_dataframe(df, n_qubits, depth, domain)
    plot_time_evolution(df_funcs, max_lines=6, outfile=str(exp_path / "time_evo.png"))
    
    # TODO. in the plots I do not see the last point t=5, why?

if __name__ == "__main__":
    main()