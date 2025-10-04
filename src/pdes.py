from __future__ import annotations


import numpy as np

from typing_extensions import Optional
from abc import ABC, abstractmethod

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

from .ansatz import HEAnsatz
from .circuit import (
    circuit_overlap,
    circuit_adder_overlap_1d,
    circuit_diag_overlap,
    circuit_nonlinear_overlap_1d,
    circuit_adder_overlap_2d,
    circuit_nonlinear_overlap_2d,
)

    
class BasePDE(ABC):
    """Base class for PDE loss functions."""
    
    @property
    def name(self) -> str:
        return self.__class__._name

    def __init__(
        self, 
        lambda0: float,
        lambdas: np.ndarray | list[float],
        tau: float,
        n_qubits: int,
        depth: int, 
    ) -> None:
        """
        Args:
            lambda0 (float): Scalar parameter for the current step (e.g., initial amplitude).
            lambdas (np.ndarray | list[float]): Variational parameters for the ansatz circuit.
            tau (float): Time step size or evolution parameter.
            n_qubits (int): Number of qubits in the quantum circuit.
            depth (int): Number of layers in the ansatz circuit.
        """
        self.lambda0 = lambda0
        self.lambdas = lambdas
        self.tau = tau
        self.n_qubits = n_qubits
        self.depth = depth

    def ancilla_z_exp(
        self,
        circ: QuantumCircuit, 
        shots: Optional[int] = None,
    ) -> float:
        """
        Compute the expectation value of the Pauli Z observable on the ancilla qubit (anc[0])
        in a Hadamard-test circuit.

        This function measures ⟨Z⟩ on the ancilla qubit, which is assumed to be the first qubit
        (index 0) and part of a register named 'anc'. The expectation value is computed either
        exactly (using the statevector simulator) or by sampling (using the qasm simulator).

        Args:
            circ (QuantumCircuit): Quantum circuit with an ancilla register named 'anc'.
            shots (Optional[int]): If None, compute the exact expectation value using the 
                                     statevector simulator. If an integer, estimate the 
                                     expectation value by sampling with the given number of shots.

        Returns:
            float: The expectation value ⟨Z⟩ on the ancilla qubit.
        """
        if shots is None:
            # Exact expectation from statevector
            backend = Aer.get_backend("statevector_simulator")
            t_qc = transpile(circ, backend)
            sv = backend.run(t_qc).result().get_statevector(t_qc)
            n = circ.num_qubits
            probs = np.abs(sv) ** 2
            p0, p1 = 0.0, 0.0
            
            for idx, p in enumerate(probs):
                anc_bit = (idx >> n-1) & 1  # TODO: implement more qiskit-like approach
                if anc_bit == 0:
                    p0 += p
                else:
                    p1 += p
            return p0 - p1
        
        else:
            # Sampling
            circ_meas = circ.copy()
            circ_meas.measure_all()
            backend = Aer.get_backend("qasm_simulator")
            t_qc = transpile(circ_meas, backend)
            counts = backend.run(t_qc, shots=shots).result().get_counts()
            # ancilla is most significant qubit (MSB) => look at first char of key
            N0, N1 = 0, 0
            for bitstring, count in counts.items():
                if bitstring[0] == '0':
                    N0 += count
                else:
                    N1 += count
            return (N0 - N1) / shots
        
    def update_state(
        self,
        lambda0: float,
        lambdas: np.ndarray | list[float],
    ) -> None:
        """
        Update the internal state of the PDE object with new parameters.

        Args:
            lambda0 (float): New scaling parameter.
            lambdas (np.ndarray | list[float]): New variational parameters for the ansatz circuit.
        """
        self.lambda0 = lambda0
        self.lambdas = lambdas

    @abstractmethod
    def cost(self, lambdas: np.ndarray | list[float]) -> float:
        pass


class Burgers1D(BasePDE):
    """
    Implements the cost function for a single Euler step of the 1D Burgers' equation
    as shown in Lubasch et al.

    The cost function (Eq. S4) is:
      C(λ0, λ) = |λ0|^2
                 - 2 Re{ λ0 · (tilde_λ0)^* <0| Ũ^† (1 + τ O) U(λ) |0> } + const

    where:
      - O = v Δ - tilde_λ0 · diag(tilde_f) ∇,
      - tilde_f = tilde_λ0 · tilde_ψ,
      - Ũ|0⟩ = tilde_ψ.

    Notes:
    - The constant term is omitted as it does not affect minimization.
    """

    def __init__(
        self, 
        lambda0: float,
        lambdas: np.ndarray | list[float],
        nu: float, 
        tau: float,
        n_qubits: int,
        depth: int, 
    ) -> None:
        """
        Initialize the Burgers1D cost function. Most parameters are initialized by the BasePDE class.

        Args:
            lambda0 (float): Scalar parameter for the current step (e.g., initial amplitude).
            lambdas (np.ndarray | list[float]): Variational parameters for the ansatz circuit.
            nu (float): Viscosity parameter for the Burgers' equation.
            tau (float): Time step size or evolution parameter.
            n_qubits (int): Number of qubits in the quantum circuit.
            depth (int): Number of layers in the ansatz circuit.
        """
        super().__init__(lambda0, lambdas, tau, n_qubits, depth)
        self.nu = nu


    def cost(self, lambdas: np.ndarray | list[float]) -> float:
        """
        Compute the variational cost function C(lambda0, lambda) for a single Euler step
        of the 1D Burgers' equation, up to an additive constant.

        This method assembles and runs a set of Hadamard-test quantum circuits to evaluate
        the required expectation values for the cost function, as described in Lubasch et al.

        Args:
            lambdas (np.ndarray | list[float]): New parameters for the current step,
                with lambdas[0] as λ0 and lambdas[1:] as the variational parameters.

        Returns:
            float: The value of the cost function (up to an additive constant).
        """
        lambda0_new = float(lambdas[0])
        lambdas_new = np.array(lambdas[1:], copy=True)  # TODO: remove copy if not needed

        # 0) build current variational circuit U_var from ansatz
        ansatz = HEAnsatz(n_qubits=self.n_qubits, depth=self.depth)
        U_tilde_circ = ansatz.qc(self.lambdas)
        U_var = ansatz.qc(lambdas_new)

        # 1) build the 5 hadamard-test circuits
        qc_w0 = circuit_overlap(U_var, U_tilde_circ)                           # w0
        qc_wA = circuit_adder_overlap_1d(U_var, U_tilde_circ, inverse=False)   # wA
        qc_wAinv = circuit_adder_overlap_1d(U_var, U_tilde_circ, inverse=True) # wA^{-1}
        qc_wD = circuit_diag_overlap(U_var, U_tilde_circ, inverse=True)        # wD
        qc_wDA = circuit_nonlinear_overlap_1d(U_var, U_tilde_circ)             # wDA

        # 2) run circuits (use exact statevector)
        # Note: ancilla_z_exp assumes ancilla is qubit index 0 (first register 'anc')
        w0 = self.ancilla_z_exp(qc_w0)
        wA = self.ancilla_z_exp(qc_wA)
        wAinv = self.ancilla_z_exp(qc_wAinv)
        wD = self.ancilla_z_exp(qc_wD)
        wDA = self.ancilla_z_exp(qc_wDA)

        # 3) assemble LapVal and NonlinVal
        LapVal = wA + wAinv - 2.0 * w0
        NonlinVal = wDA - wD

        # 4) scalar s = Re <tildeψ|(1+τ O)|ψ>
        s = w0 + self.tau * (self.nu * LapVal - self.lambda0 * NonlinVal)

        # 5) cost up to const: |λ0|^2 - 2 Re{ λ0 (tilde_λ0)^* s }
        cost = (lambda0_new ** 2) - 2.0 * np.real(lambda0_new * np.conj(self.lambda0)) * s

        return cost


class Burgers2D(Burgers1D):
    """
    Implements the cost function for a single Euler step of the 2D Burgers' equation
    following the method outlined in Lubasch et al.

    The cost function (Eq. S4) is:
      C(λ0, λ) = |λ0|^2
                 - 2 Re{ λ0 · (tilde_λ0)^* <0| Ũ^† (1 + τ O) U(λ) |0> } + const

    where:
      - O = v Δ - tilde_λ0 · diag(tilde_f) ∇,
      - tilde_f = tilde_λ0 · tilde_ψ,
      - Ũ|0⟩ = tilde_ψ.

    Notes:
    - The constant term is omitted as it does not affect minimization.
    """

    def cost(self, lambdas):
        """
        Compute the variational cost function C(lambda0, lambda) for a single Euler step
        of the 2D Burgers' equation, up to an additive constant.

        This method assembles and runs a set of Hadamard-test quantum circuits to evaluate
        the required expectation values for the cost function, as described in Lubasch et al.

        Args:
            lambdas (np.ndarray | list[float]): New parameters for the current step,
                with lambdas[0] as λ0 and lambdas[1:] as the variational parameters.

        Returns:
            float: The value of the cost function (up to an additive constant).
        """
        lambda0_new = float(lambdas[0])
        lambdas_new = np.array(lambdas[1:], copy=True)  # TODO: remove copy if not needed

        # 0) build current variational circuit U_var from ansatz
        ansatz = HEAnsatz(n_qubits=self.n_qubits, depth=self.depth)
        U_tilde_circ = ansatz.qc(self.lambdas)
        U_var = ansatz.qc(lambdas_new)

        # 1) build the hadamard-test circuits
        qc_w0 = circuit_overlap(U_var, U_tilde_circ)                                   
        qc_wA_x, qc_wA_y = circuit_adder_overlap_2d(U_var, U_tilde_circ, inverse=False)
        qc_wAinv_x, qc_wAinv_y = circuit_adder_overlap_2d(U_var, U_tilde_circ, inverse=True)
        qc_wD = circuit_diag_overlap(U_var, U_tilde_circ, inverse=True) 
        qc_wDA_x, qc_wDA_y = circuit_nonlinear_overlap_2d(U_var, U_tilde_circ)      

        # 2) run circuits (use exact statevector)
        # Note: ancilla_z_exp assumes ancilla is qubit index 0 (first register 'anc')
        w0 = self.ancilla_z_exp(qc_w0)
        wA_x, wA_y = self.ancilla_z_exp(qc_wA_x), self.ancilla_z_exp(qc_wA_y)
        wAinv_x, wAinv_y = self.ancilla_z_exp(qc_wAinv_x), self.ancilla_z_exp(qc_wAinv_y)
        wD = self.ancilla_z_exp(qc_wD)
        wDA_x, wDA_y = self.ancilla_z_exp(qc_wDA_x), self.ancilla_z_exp(qc_wDA_y)

        # 3) assemble LapVal and NonlinVal
        LapVal = (wA_x + wAinv_x + wA_y + wAinv_y) - 4.0 * w0
        NonlinVal = (wDA_x - wD) + (wDA_y - wD)

        # 4) scalar s = Re <tildeψ|(1+τ O)|ψ>
        s = w0 + self.tau * (self.nu * LapVal - self.lambda0 * NonlinVal)

        # 5) cost up to const: |λ0|^2 - 2 Re{ λ0 (tilde_λ0)^* s }
        cost = (lambda0_new ** 2) - 2.0 * np.real(lambda0_new * np.conj(self.lambda0)) * s

        return cost
    

class Diffusion1D(BasePDE):
    """
    Implements the cost function for a single Euler step of the 1D Diffusion equation.

    The cost function is:
      C(λ0, λ) = |λ0|^2
                 
    """

    def __init__(
        self, 
        lambda0: float,
        lambdas: np.ndarray | list[float],
        D: float, 
        tau: float,
        n_qubits: int,
        depth: int, 
    ) -> None:
        """
        Initialize the Diffusion1D cost function. Most parameters are initialized by the BasePDE class.

        Args:
            lambda0 (float): Scalar parameter for the current step (e.g., initial amplitude).
            lambdas (np.ndarray | list[float]): Variational parameters for the ansatz circuit.
            D (float): Diffusion coefficient.
            tau (float): Time step size or evolution parameter.
            n_qubits (int): Number of qubits in the quantum circuit.
            depth (int): Number of layers in the ansatz circuit.
        """
        super().__init__(lambda0, lambdas, tau, n_qubits, depth)
        self.D = D


    def cost(self, lambdas: np.ndarray | list[float]) -> float:
        """
        """
        pass


class Diffusion2D(Diffusion1D):
    """
    Implements the cost function for a single Euler step of the 2D Diffusion equation.
    """
    
    def cost(self, lambdas: np.ndarray | list[float]) -> float:
        """
        """
        pass    