import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from qiskit.quantum_info import Statevector
from src.variational import SingleParameterAnsatz

# TODO: this is all 1D, we want 2D eventually

def reconstruct_function(lambda0: float,
                         lam: float,
                         n_qubits: int,
                         depth: int,
                         domain: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct f(x) from parameters (lambda0, lambda).
    - Build SingleParameterAnsatz with 'lam'
    - Get statevector ψ (size N = 2**n_qubits)
    - Map basis index k to x_k in [domain[0], domain[1]) uniformly (endpoint=True)
    - Return x, f(x) = lambda0 * ψ_k (real part), and ψ (complex for reference)
    """
    N = 2 ** n_qubits
    x = np.linspace(domain[0], domain[1], N, endpoint=True)

    qc = SingleParameterAnsatz(n_qubits=n_qubits, depth=depth).qc(lam)
    psi = Statevector.from_instruction(qc).data  # shape (N,)

    # With Ry-only ansatz and CNOTs, amplitudes are real; be explicit:
    f = lambda0 * np.real(psi)
    return x, f, psi


def time_evolution_dataframe(df_params: pd.DataFrame,
                             n_qubits: int,
                             depth: int,
                             domain: list[float]) -> pd.DataFrame:
    """
    From a dataframe of per-step parameters (time, lambda0, lambda),
    build a long-form dataframe with columns: time, x, f, psi.
    """
    rows = []
    df_params = df_params.sort_values("time")
    for _, row in df_params.iterrows():
        x, f, psi = reconstruct_function(
            lambda0=float(row["lambda0"]),
            lam=float(row["lambda"]),
            n_qubits=n_qubits,
            depth=depth,
            domain=domain,
        )
        rows.append(pd.DataFrame({
            "time": row["time"],
            "x": x,
            "f": f,
            "psi": psi
        }))
    return pd.concat(rows, ignore_index=True)

# TODO: remove max _lines, you still want last and first line
def plot_time_evolution(df_funcs: pd.DataFrame,
                        max_lines: int = 6,
                        outfile: str = "exp_results/burgers_evolution.png",
                        base_color: str = "blue",
) -> None:
    """
    Plot f(x,t) at a handful of times to visualize evolution.
    - Picks up to 'max_lines' evenly spaced times.
    """
    times_all = sorted(df_funcs["time"].unique())
    if max_lines is not None and len(times_all) > max_lines:
        idx = np.linspace(0, len(times_all) - 1, max_lines, dtype=int)
        times = [times_all[i] for i in idx]
    else:
        times = times_all

    # Ensure output directory
    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Normalize for color intensity
    n = len(times)
    for i, t in enumerate(times):
        sub = df_funcs[df_funcs["time"] == t]
        # Intensity: from 0.2 (dim) to 1.0 (full color)
        intensity = 0.2 + 0.8 * (i / (n - 1) if n > 1 else 1)
        color = plt.get_cmap("Blues")(intensity) if base_color == "blue" else plt.get_cmap("Reds")(intensity)
        plt.plot(sub["x"].to_numpy(), sub["f"].to_numpy(), lw=2, label=f"t={t:.3f}", color=color)

    plt.xlabel("x")
    plt.ylabel("f(x,t)")
    plt.title("Time evolution of variational solution f(x,t)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved time evolution plot to {out_path.resolve()}")