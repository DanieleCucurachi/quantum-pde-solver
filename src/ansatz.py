from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from qiskit import QuantumCircuit


class BaseAnsatz(ABC):
    """Variational ansatz base class."""

    def __init__(self, n_qubits: int, depth: int) -> None:
        """
        Initialize ansatz.
        
        Args:
            n_qubits (int): Number of qubits
            depth (int): Number of layers
        """
        self.n_qubits = n_qubits
        self.depth = depth
    
    @abstractmethod
    def qc(self, params: np.ndarray | list[float]) -> QuantumCircuit:
        """Create quantum circuit for given parameters."""
        pass
    

class HEAnsatz(BaseAnsatz):
    """Hardware-efficient ansatz.
    
    Structure: depth layers of Ry rotations + CNOT chain entanglers.
    """
    
    def __init__(self, n_qubits: int, depth: int) -> None:
        """
        Initialize ansatz.
        
        Args:
            n_qubits (int): Number of qubits
            depth (int): Number of layers
        """
        super().__init__(n_qubits, depth)
        self.n_params = n_qubits * depth
    
    def qc(self, params: np.ndarray | list[float]) -> QuantumCircuit:
        """
        Create hardware-efficient ansatz circuit.
        
        Args:
            params (np.ndarray | list[float]): Parameter array (flat or shape (depth, n_qubits))
            
        Returns:
            QuantumCircuit
        """

        # Convert params to np.ndarray if it's a list or similar
        params = np.array(params)
        
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


class SingleParameterAnsatz(BaseAnsatz):
    """
    Single-parameter ansatz where all Ry gates use the same parameter λ.
    """
    
    def __init__(self, n_qubits: int, depth: int) -> None:
        """
        Initialize ansatz.
        
        Args:
            n_qubits (int): Number of qubits
            depth (int): Number of layers
        """
        super().__init__(n_qubits, depth)
        self.n_params = n_qubits * depth
    
    def qc(self, lambda_param: np.ndarray | list[float]) -> QuantumCircuit:
        """
        Create single-parameter ansatz circuit.
        
        Args:
            lambda_param (np.ndarray | list[float]): Single parameter λ for all Ry gates
            
        Returns:
            QuantumCircuit
        """

        lambda_param = float(lambda_param[0])
        
        qc = QuantumCircuit(self.n_qubits)
        
        for layer in range(self.depth):
            # Same Ry(λ) on all qubits
            for q in range(self.n_qubits):
                qc.ry(lambda_param, q)
            
            # CNOT chain
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
        
        return qc
    
    def random_param(self) -> list[float]:
        """Generate random parameter in [-π, π]."""
        return [np.random.uniform(-np.pi, np.pi)]
