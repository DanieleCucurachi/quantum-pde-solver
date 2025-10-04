import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from qiskit.quantum_info import Statevector
from src.variational import SingleParameterAnsatz, HEAnsatz


def reconstruct_function(lambda0: float,
                         lambdas: np.ndarray | list[float],
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

    qc = HEAnsatz(n_qubits=n_qubits, depth=depth).qc(lambdas)
    psi = Statevector.from_instruction(qc).data  # shape (N,)

    # With Ry-only ansatz and CNOTs, amplitudes are real; be explicit:
    f = lambda0 * np.real(psi)
    return x, f, psi


# 1D

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
            lambdas=row["lambdas"],
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
                        max_lines: int = 10,
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


# 2D

def time_evolution_dataframe_2d(
    df_params: pd.DataFrame,
    n_qubits: int,
    depth: int,
    domain: list[tuple[float, float]],
) -> pd.DataFrame:
    """
    From a dataframe of per-step parameters (time, lambda0, lambda),
    build a long-form dataframe with columns: time, x, y, f, psi.
    The 1D vector is mapped to a 2D grid as described.
    """
    rows = []
    df_params = df_params.sort_values("time")
    L = 2 ** n_qubits
    n_side = int(np.sqrt(L))
    assert n_side ** 2 == L, "Number of basis states must be a perfect square for 2D."
    (xmin, xmax), (ymin, ymax) = domain
    xs = np.linspace(xmin, xmax, n_side, endpoint=True)
    ys = np.linspace(ymin, ymax, n_side, endpoint=True)
    for _, row in df_params.iterrows():
        _, f, psi = reconstruct_function(
            lambda0=float(row["lambda0"]),
            lambdas=row["lambdas"],
            n_qubits=n_qubits,
            depth=depth,
            domain=[(xmin, xmax), (ymin, ymax)],
        )
        # Map 1D vector to 2D grid
        for idx in range(L):
            # TODO: amke this a utility function
            bin_str = format(idx, f'0{n_qubits}b')
            x_idx = int(bin_str[:n_qubits // 2], 2)
            y_idx = int(bin_str[n_qubits // 2:], 2)
            rows.append({
                "time": row["time"],
                "x": xs[x_idx],
                "y": ys[y_idx],
                "f": f[idx],
                "psi": psi[idx]
            })
    return pd.DataFrame(rows)

def plot_time_evolution_2d(
    df_funcs: pd.DataFrame,
    max_plots: int = 16,
    outfile: str = "exp_results/burgers_evolution_2d.png",
    cmap: str = "viridis"
) -> None:
    """
    For each selected time, plot f(x, y, t) as a 2D heatmap in a subplot.
    """
    times_all = sorted(df_funcs["time"].unique())
    if max_plots is not None and len(times_all) > max_plots:
        idx = np.linspace(0, len(times_all) - 1, max_plots, dtype=int)
        times = [times_all[i] for i in idx]
    else:
        times = times_all

    n_plots = len(times)
    ncols = min(n_plots, 3)
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for i, t in enumerate(times):
        sub = df_funcs[df_funcs["time"] == t]
        xs = np.sort(sub["x"].unique())
        ys = np.sort(sub["y"].unique())
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        F = sub.pivot(index="x", columns="y", values="f").values

        ax = axes[i // ncols, i % ncols]
        im = ax.pcolormesh(X, Y, F, cmap=cmap, shading="auto")
        ax.set_title(f"t={t:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)

    # Hide unused subplots
    for j in range(i + 1, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')

    plt.tight_layout()
    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved 2D time evolution plot to {out_path.resolve()}")