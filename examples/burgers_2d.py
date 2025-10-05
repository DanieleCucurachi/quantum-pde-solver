from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import set_seeds, gaussian_state, fix_lambda0_sign
from src.ansatz import HEAnsatz
from src.pdes import Burgers2D
from src.plot import time_evolution_dataframe_2d, plot_time_evolution_2d
from src.time_evo import (
    prepare_initial_state,
    run_time_evolution,
)


def parse_args():
    parser = argparse.ArgumentParser(description="2D Burgers Simulation")
    # Circuit/ansatz parameters
    parser.add_argument("--n_qubits", type=int, default=4, help="Number of qubits (must be even for 2D)")
    parser.add_argument("--depth", type=int, default=2, help="Ansatz circuit depth")
    # Training/simulation parameters
    parser.add_argument("--tau", type=float, default=0.1, help="Euler time step")
    parser.add_argument("--nu", type=float, default=0.1, help="Viscosity")
    parser.add_argument("--tmax", type=float, default=5.0, help="Total simulation time")
    parser.add_argument("--sigma", type=float, default=0.15, help="Initial Gaussian width")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="examples/exp_results/burgers", help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seeds(args.seed, verbose=True)

    domain = [(0.0, 1.0), (0.0, 1.0)]
    nsteps = int(args.tmax / args.tau)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "data_2d.csv"

    print("\n------ Initial State Preparation ------\n")
    target = gaussian_state(args.n_qubits, domain=domain, sigma=args.sigma)
    lambdas, init_fidelity = prepare_initial_state(
        n_qubits=args.n_qubits,
        depth=args.depth,
        target=target,
    )
    lambda0 = 1.0  # Initial scaling factor
    lambda0 = fix_lambda0_sign(HEAnsatz(args.n_qubits, args.depth), lambdas, lambda0, target)
    print("Optimal λ parameters:\n- λ0:", lambda0, "\n- λ:", lambdas)
    print("Final fidelity:", init_fidelity)

    print("\n------ Time Evolution ------\n")
    df = run_time_evolution(
        lambda0=lambda0,
        lambdas=lambdas,
        pde=Burgers2D(lambda0, lambdas, args.nu, args.tau, args.n_qubits, args.depth),
        nsteps=nsteps,
    )
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path.resolve()}")

    print("\n------ Results Processing ------\n")
    df_funcs = time_evolution_dataframe_2d(df, args.n_qubits, args.depth, domain)
    plot_time_evolution_2d(df_funcs, max_plots = 6, outfile=str(outdir / "time_evo_2d.png"))

if __name__ == "__main__":
    main()