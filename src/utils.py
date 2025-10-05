from __future__ import annotations

import os
import random

import numpy as np

from qiskit.quantum_info import Statevector
from qiskit import transpile
from qiskit_aer import Aer, AerSimulator

from .ansatz import BaseAnsatz, HEAnsatz


def set_seeds(
    seed: int, 
    verbose: bool = False,
) -> None:
    """
    Minimal seeding for experiment reproducibility: Python random, NumPy, and hash seed.

    Args:
        seed (int): The random seed to use.
        verbose (bool): If True, print the seed settings.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if verbose:
        print(f"[seed] random.seed={seed}")
        print(f"[seed] np.random.seed={seed}")
        print(f"[seed] PYTHONHASHSEED={os.environ['PYTHONHASHSEED']}")


def amplitude_encoding_2d(idx: int, n_qubits: int) -> tuple[int, int]:
    """
    Convert a linear index to 2D grid indices (x_idx, y_idx) using binary encoding.

    Example:
        For n_qubits=4 (N=16), the basis states are indexed 0 to 15.
        The mapping is:
            idx | bin  | x_idx | y_idx
            ----|------|-------|------
                0  | 0000 |   0   |  0
                1  | 0001 |   0   |  1
                2  | 0010 |   0   |  2
                3  | 0011 |   0   |  3
                4  | 0100 |   1   |  0
                 ...
                15 | 1111 |   3   |  3

    Args:
        idx (int): Linear index (0 to 2**n_qubits - 1).
        n_qubits (int): Total number of qubits (must be even for 2D).
    
    Returns:
        tuple[int, int]: (x_idx, y_idx) grid indices.
    """
    # Binary encoding: first half for x, second half for y
    bin_str = format(idx, f'0{n_qubits}b')  # MSB right, LSB left 
    x_idx = int(bin_str[:n_qubits // 2], 2)  # int(a, b) converts num a to base b 
    y_idx = int(bin_str[n_qubits // 2:], 2)
    return x_idx, y_idx


def gaussian_state(
    n_qubits: int, 
    domain: list[tuple], 
    sigma=0.15
) -> np.ndarray:
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
            x_idx, y_idx = amplitude_encoding_2d(idx, n_qubits)
            x = xs[x_idx]
            y = ys[y_idx]
            f[idx] = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * sigma ** 2))

    else:
        raise NotImplementedError("Only 1D and 2D domains are supported.")

    # Normalize
    f = f / np.linalg.norm(f)

    return f

# TODO: move to prepare initial state | find better solution
def fix_lambda0_sign(
    ansatz: BaseAnsatz, 
    lambdas: np.ndarray, 
    lambda0: float, 
    target: np.ndarray,
) -> float:
    """
    Ensure the output of the quantum circuit matches the sign of the target state.
    If the overlap is negative, flip the sign of lambda0.

    Args:
        ansatz (BaseAnsatz): Ansatz object with .qc(params) method.
        params (np.ndarray): Optimized ansatz parameters.
        lambda0 (float): Scaling parameter.
        target (np.ndarray): Target statevector (numpy array).

    Returns:
        float: The fixed lambda0 value.
    """

    qc = ansatz.qc(lambdas)
    psi = Statevector.from_instruction(qc).data
    overlap = np.vdot(target, psi)
    if np.real(overlap) < 0:
        lambda0 = -lambda0
    return lambda0


def reconstruct_function(
    lambda0: float,
    lambdas: np.ndarray | list[float],
    n_qubits: int,
    depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct the real-space solution f(x) from quantum circuit parameters in 1D.

    This function:
      - Builds a hardware-efficient ansatz circuit (HEAnsatz) using the provided parameters.
      - Simulates the circuit to obtain the quantum statevector ψ of size N = 2**n_qubits.
      - Computes the real-valued function f(x) = lambda0 * Re(ψ_k) at each grid point x_k,
        where k is the basis index.
      - Returns both the reconstructed function values and the full complex statevector.

    Args:
        lambda0 (float): Scalar prefactor for the reconstructed function.
        lambdas (np.ndarray | list[float]): Ansatz parameters for the quantum circuit.
        n_qubits (int): Number of qubits (defines grid size).
        depth (int): Number of ansatz layers.

    Returns:
        tuple:
            f (np.ndarray): Real-valued function values at each grid point, shape (N,).
            psi (np.ndarray): Full complex statevector amplitudes, shape (N,).
    """
    
    qc = HEAnsatz(n_qubits=n_qubits, depth=depth).qc(lambdas)
    psi = Statevector.from_instruction(qc).data  # shape (N,)
 
    f = lambda0 * np.real(psi)
    return f, psi


def fidelity(
    params: np.ndarray, 
    ansatz: BaseAnsatz, 
    target_state: np.ndarray,
) -> float:
    """
    Compute the fidelity between the quantum state prepared by the ansatz circuit
    (with given parameters) and a target state.

    It is defined as |⟨target_state|psi⟩|^2, where |psi⟩ is the statevector
    output by the ansatz circuit. The negative value is returned for compatibility
    with minimizers (so that maximizing fidelity becomes a minimization problem).

    Args:
        params: Parameters for the ansatz circuit.
        ansatz: Ansatz object with a .qc(params) method returning a QuantumCircuit.
        target_state: Target statevector as a 1D numpy array.

    Returns:
        float: Negative fidelity value.
    """
    qc = ansatz.qc(params)
    backend = Aer.get_backend("statevector_simulator")
    sv = Statevector.from_instruction(transpile(qc, backend))
    psi = sv.data
    return -np.abs(np.vdot(target_state, psi))**2  # negative for minimizer compatibility


def make_seeded_aer_simulator(seed: int) -> AerSimulator:
    """
    Create a Qiskit AerSimulator with a fixed random seed.
    Requires qiskit-aer to be installed.

    Args:
        seed (int): The random seed to use for the simulator.

    Returns:
        AerSimulator: An AerSimulator instance with the specified seed.
    """
    sim = AerSimulator(seed_simulator=seed)
    sim.set_options(seed_simulator=seed)
    return sim