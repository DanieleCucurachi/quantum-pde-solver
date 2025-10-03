from __future__ import annotations

import os
import random
import numpy as np

from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator


# --- target Gaussian in 1D ---
# TODO: add domani here!!!
def gaussian_state(n_qubits: int, sigma=0.15):
    L = 2**n_qubits
    xs = np.linspace(0, 1, L, endpoint=False)  # grid
    f = np.exp(-0.5 * ((xs - 0.5)/sigma)**2)
    f = f / np.linalg.norm(f)  # normalize
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