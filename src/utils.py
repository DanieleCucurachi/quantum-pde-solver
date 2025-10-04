from __future__ import annotations

import os
import random
import numpy as np

from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator


def gaussian_state(n_qubits: int, domain: list[tuple], sigma=0.15) -> np.ndarray:
    """
    Generate a normalized 1D or 2D Gaussian state vector on a grid of length 2**n_qubits.

    Args:
        n_qubits (int): Number of qubits (defines grid size).
        domain (list[tuple]): List of tuples specifying the domain for each dimension.
            - For 1D: [(xmin, xmax)]
            - For 2D: [(xmin, xmax), (ymin, ymax)]
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: Normalized Gaussian vector of length 2**n_qubits.

    Raises:
        NotImplementedError: If domain has more than 2 dimensions.
    """
    L = 2 ** n_qubits

    if domain is None or len(domain) == 1:
        # 1D case
        if domain is not None:
            xmin, xmax = domain[0]
            xs = np.linspace(xmin, xmax, L, endpoint=False)
        else:
            xs = np.linspace(0, 1, L, endpoint=False)
        f = np.exp(-0.5 * ((xs - 0.5) / sigma) ** 2)
    elif len(domain) == 2:
        # 2D case
        n_side = int(np.sqrt(L))
        if n_side ** 2 != L:
            raise ValueError("For 2D, 2**n_qubits must be a perfect square.")
        (xmin, xmax), (ymin, ymax) = domain
        xs = np.linspace(xmin, xmax, n_side, endpoint=False)
        ys = np.linspace(ymin, ymax, n_side, endpoint=False)
        f = np.zeros(L)
        for idx in range(L):
            # Binary encoding: first half for x, second half for y
            bin_str = format(idx, f'0{n_qubits}b')  # MSB right, LSB left 
            x_idx = int(bin_str[:n_qubits // 2], 2)  # int(a, b) converts a to base b 
            y_idx = int(bin_str[n_qubits // 2:], 2)
            x = xs[x_idx]
            y = ys[y_idx]
            f[idx] = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * sigma ** 2))
    else:
        raise NotImplementedError("Only 1D and 2D domains are supported.")

    f = f / np.linalg.norm(f)
    return f


# --- fidelity ---
def fidelity(params, ansatz, target_state):
    qc = ansatz.qc(params)  # TODO: this will have to be changed for multiple params
    backend = Aer.get_backend("statevector_simulator")
    sv = Statevector.from_instruction(transpile(qc, backend))
    psi = sv.data
    return -np.abs(np.vdot(target_state, psi))**2  # negative for minimizer  # TODO: amke it positive and move -


def set_seeds(seed: int, verbose: bool = False) -> None:
    """
    Minimal seeding for experiment reproducibility: Python random, NumPy, and hash seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if verbose:
        print(f"[seed] random.seed={seed}")
        print(f"[seed] np.random.seed={seed}")
        print(f"[seed] PYTHONHASHSEED={os.environ['PYTHONHASHSEED']}")


def make_seeded_aer_simulator(seed: int) -> AerSimulator:
    """
    Create a Qiskit AerSimulator with a fixed random seed.
    Requires qiskit-aer to be installed.
    """
    sim = AerSimulator(seed_simulator=seed)
    sim.set_options(seed_simulator=seed)
    return sim


# # Set random seed for reproducibility
# qiskit_algorithms.utils.algorithm_globals.random_seed = self.random_seed
# # TODO: include qiskit seed
# try:
#     backend.simulator.options.seed_simulator = self.random_seed
# except:
#     # If the backend does not have a simulator, we ignore this setting.
#     pass