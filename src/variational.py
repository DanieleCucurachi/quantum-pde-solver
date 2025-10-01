import numpy as np
from qiskit import QuantumCircuit
from typing import Optional

# TODO: use parametrized circuits
class Ansatz:
    """va
    Hardware-efficient variational ansatz.
    
    Structure: depth layers of Ry rotations + CNOT chain entanglers.
    """
    
    def __init__(self, n_qubits: int, depth: int):
        """
        Initialize ansatz.
        
        Args:
            n_qubits: Number of qubits
            depth: Number of layers
        """
        self.n_qubits = n_qubits
        self.depth = depth
        self.n_params = n_qubits * depth
    
    def qc(self, params: np.ndarray) -> QuantumCircuit:
        """
        Create hardware-efficient ansatz circuit.
        
        Args:
            params: Parameter array (flat or shape (depth, n_qubits))
            
        Returns:
            Quantum circuit
        """
        
        # Reshape params if needed
        if params.ndim == 1:
            params = params.reshape(self.depth, self.n_qubits)
        
        qc = QuantumCircuit(self.n_qubits)
        
        for layer in range(self.depth):
            # Ry rotations on each qubit
            for q in range(self.n_qubits):
                qc.ry(params[layer, q], q)
            
            # CNOT chain: 0->1->2->...->n-1
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
        
        return qc
    
    def random_params(self) -> np.ndarray:
        """Generate random parameters in [-π, π]."""
        return np.random.uniform(-np.pi, np.pi, self.n_params)


class SingleParameterAnsatz:
    """
    Single-parameter ansatz where all Ry gates use the same parameter λ.
    """
    
    def __init__(self, n_qubits: int, depth: int):
        self.n_qubits = n_qubits
        self.depth = depth
        self.n_params = 1
    
    # TODO: find better solution to account for lambda
    def qc(self, lambda_param: float) -> QuantumCircuit:
        """
        Create single-parameter ansatz circuit.
        
        Args:
            lambda_param: Single parameter λ for all Ry gates
            
        Returns:
            Quantum circuit
        """

        if isinstance(lambda_param, (list, np.ndarray)):
            lambda_param = float(lambda_param[0])
        else:
            lambda_param = float(lambda_param)
        
        qc = QuantumCircuit(self.n_qubits)
        
        for layer in range(self.depth):
            # Same Ry(λ) on all qubits
            for q in range(self.n_qubits):
                qc.ry(lambda_param, q)
            
            # CNOT chain
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
        
        return qc
    
    def random_param(self) -> float:
        """Generate random parameter in [-π, π]."""
        return np.random.uniform(-np.pi, np.pi)
