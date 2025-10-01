from __future__ import annotations

# TODO: update the training logic so that qc are not created everytime from scratch

from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
import numpy as np

class Ansatz:
    """
    Hardware-efficient variational ansatz with symbolic parameters.
    Structure: depth layers of Ry rotations + CNOT chain entanglers.
    """
    
    def __init__(self, n_qubits: int, depth: int):
        self.n_qubits = n_qubits
        self.depth = depth
        self.n_params = n_qubits * depth
        self.params = ParameterVector("lambda", self.n_params)
        
        # build circuit ONCE with symbolic parameters
        self._qc = QuantumCircuit(self.n_qubits)
        idx = 0
        for layer in range(self.depth):
            for q in range(self.n_qubits):
                self._qc.ry(self.params[idx], q)
                idx += 1
            for q in range(self.n_qubits - 1):
                self._qc.cx(q, q + 1)

    @property
    def qc(self) -> QuantumCircuit:
        """Return the parameterized circuit (symbolic)."""
        return self._qc

    def bind(self, values: np.ndarray) -> QuantumCircuit:
        """Bind parameter values without rebuilding the circuit."""
        return self._qc.bind_parameters({self.params[i]: values[i] for i in range(self.n_params)})


class SingleParameterAnsatz:
    """
    Single-parameter ansatz with symbolic parameter λ.
    """
    def __init__(self, n_qubits: int, depth: int):
        self.n_qubits = n_qubits
        self.depth = depth
        self.param = Parameter("lambda")
        
        # build symbolic circuit ONCE
        self._qc = QuantumCircuit(self.n_qubits)
        for layer in range(self.depth):
            for q in range(self.n_qubits):
                self._qc.ry(self.param, q)
            for q in range(self.n_qubits - 1):
                self._qc.cx(q, q + 1)

    @property
    def qc(self) -> QuantumCircuit:
        """Return the parameterized circuit (symbolic)."""
        return self._qc

    def bind(self, value: float) -> QuantumCircuit:
        """Bind λ to a specific float value."""
        return self._qc.bind_parameters({self.param: value})
