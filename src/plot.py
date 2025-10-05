from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import amplitude_encoding_2d, reconstruct_function


# 1D

def time_evolution_dataframe_1d(
    df_params: pd.DataFrame,
    n_qubits: int,
    depth: int,
    domain: list[tuple[float, float]]
) -> pd.DataFrame:
    """
    Construct a long-form DataFrame representing the time evolution of the variational solution
    in 1D.

    For each row in the input DataFrame (containing per-step parameters: time, lambda0, lambdas),
    this function:
      - Reconstructs the quantum state and the corresponding real-valued function f(x) at each 
        time step, using the provided quantum circuit parameters.
      - Maps each basis index to a spatial grid point x in the interval [domain[0][0], domain[0][1]].
      - Collects the results into a long-form DataFrame with columns:
            - time: time step value
            - x: spatial grid point
            - f: real-valued function at x (from the quantum state)
            - psi: complex statevector amplitude at x

    Args:
        df_params (pd.DataFrame): DataFrame with columns ["time", "lambda0", "lambdas"] for each
                                  time step.
        n_qubits (int): Number of qubits (defines grid size).
        depth (int): Number of ansatz layers.
        domain (list[float]): [xmin, xmax] interval for the spatial grid.

    Returns:
        pd.DataFrame: Long-form DataFrame with columns ["time", "x", "f", "psi"] for all time steps.
    """
    rows = []

    N = 2 ** n_qubits
    x = np.linspace(domain[0][0], domain[0][1], N, endpoint=True)
    
    df_params = df_params.sort_values("time")
    
    for _, row in df_params.iterrows():
        f, psi = reconstruct_function(
            lambda0=float(row["lambda0"]),
            lambdas=row["lambdas"],
            n_qubits=n_qubits,
            depth=depth,
        )
        rows.append(pd.DataFrame({
            "time": row["time"],
            "x": x,
            "f": f,
            "psi": psi
        }))
    return pd.concat(rows, ignore_index=True)


def plot_time_evolution_1d(
    df_funcs: pd.DataFrame,
    max_lines: int = 5,
    outfile: str = "exp_results/time_evolution.png",
    base_color: str = "blue",
) -> None:
    """
    Plot the time evolution of the variational solution f(x, t) in 1D.

    For a given DataFrame containing columns ["time", "x", "f"], this function:
      - Selects up to 'max_lines' time steps, evenly spaced throughout the simulation.
      - Plots f(x, t) as a line for each selected time step, with color intensity indicating 
        time order.
      - Saves the resulting plot to the specified output file.

    Args:
        df_funcs (pd.DataFrame): Long-form DataFrame with columns ["time", "x", "f"] (and 
                                 optionally "psi").
        max_lines (int): Maximum number of time steps to plot (evenly spaced). If None, plot all.
        outfile (str): Path to save the output plot image.
        base_color (str): Base color for the plot lines ("blue" or "red").
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
    if outfile is not None:
        out_path = Path(outfile)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved time evolution plot to {out_path.resolve()}")
    plt.show()


# 2D

def time_evolution_dataframe_2d(
    df_params: pd.DataFrame,
    n_qubits: int,
    depth: int,
    domain: list[tuple[float, float]],
) -> pd.DataFrame:
    """
    Construct a long-form DataFrame representing the time evolution of the variational solution 
    in 2D.

    For each row in the input DataFrame (containing per-step parameters: time, lambda0, lambdas),
    this function:
      - Reconstructs the quantum state and the corresponding real-valued function f(x, y) 
        at each time step, using the provided quantum circuit parameters.
      - Maps each basis index of the 1D statevector to a 2D grid point (x, y) by interpreting the 
        binary representation of the index: the first half of the bits encodes the x index, the 
        second half the y index.
      - The spatial grid is defined by the domain argument, which should be a list of two (min, max) 
        tuples, one for x and one for y.
      - Collects the results into a long-form DataFrame with columns:
            - time: time step value
            - x: spatial grid point in x
            - y: spatial grid point in y
            - f: real-valued function at (x, y) (from the quantum state)
            - psi: complex statevector amplitude at (x, y)

    Args:
        df_params (pd.DataFrame): DataFrame with columns ["time", "lambda0", "lambdas"] for each 
                                  time step.
        n_qubits (int): Number of qubits (defines grid size; must be even for 2D).
        depth (int): Number of ansatz layers.
        domain (list[tuple[float, float]]): [(xmin, xmax), (ymin, ymax)] intervals for the spatial
                                            grid.

    Returns:
        pd.DataFrame: Long-form DataFrame with columns ["time", "x", "y", "f", "psi"] for all 
                      time steps.
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
        f, psi = reconstruct_function(
            lambda0=float(row["lambda0"]),
            lambdas=row["lambdas"],
            n_qubits=n_qubits,
            depth=depth,
        )
        # Map 1D vector to 2D grid
        for idx in range(L):
            x_idx, y_idx = amplitude_encoding_2d(idx, n_qubits)
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
    cmap: str = "viridis",
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
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        fig.colorbar(im, ax=ax, label="f(x, y, t)")

    # Hide unused subplots
    for j in range(i + 1, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')

    plt.tight_layout()
    if outfile is not None:
        out_path = Path(outfile)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved 2D time evolution plot to {out_path.resolve()}")
    plt.show()