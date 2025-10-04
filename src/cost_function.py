from __future__ import annotations

import numpy as np

from typing_extensions import Optional
from abc import ABC, abstractmethod

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer

from .ansatz import SingleParameterAnsatz, HEAnsatz

# TODO: account for arbitrary n of qubits, now that you know each building block it should be easy
# TODO: refacotr into a pythonic class

# TODO. ce l'ho! base class 1D e poi la 2D sovrascrivi i metodi, e lo chimai circuit builder

def circuit_overlap(
    U_var: QuantumCircuit, 
    U_tilde: QuantumCircuit,
) -> QuantumCircuit:
    """
    Implements controlled U_var followed by controlled U_tilde^\dagger, as shown in Fig. S3(a)
    of Lubasch et al., i.e. Hadamard-test for <ψ~|ψ>.

    Args:
        U_var (QuantumCircuit): Quantum circuit preparing |ψ⟩ on n qubits.
        U_tilde (QuantumCircuit): Quantum circuit preparing |ψ̃⟩ on n qubits.

    Returns:
        QuantumCircuit: The Hadamard test circuit with (n+1) qubits (1 ancilla + n system).
    """

    n = U_var.num_qubits
    anc = QuantumRegister(1, 'anc')
    sys = QuantumRegister(n, 'sys')
    qc = QuantumCircuit(anc, sys)

    # Hadamard on ancilla
    qc.h(anc[0])

    # Controlled U_var
    U_gate = U_var.to_gate(label="U_var").control(1)
    qc.append(U_gate, [anc[0]] + list(sys))

    # Controlled U_tilde^\dagger
    U_tilde_gate = U_tilde.to_gate(label="U_tilde")
    U_tilde_dag_gate = U_tilde_gate.inverse()
    U_tilde_dag_gate.label = "U_tilde_dag"
    qc.append(U_tilde_dag_gate.control(1), [anc[0]] + list(sys))

    # Final Hadamard on ancilla
    qc.h(anc[0])

    return qc


def build_adder_block(
    anc: QuantumRegister, 
    sys: QuantumRegister,
) -> QuantumCircuit:
    """
    Build the adder QNPU block. Returns a quantum circuit implementing the adder operation A
    as descbed in Lubasch et al.

    Args:
        anc (QuantumRegister): Ancilla qubit register (1 qubit).
        sys (QuantumRegister): System qubit register (n qubits).

    Returns:
        QuantumCircuit: The adder block circuit.
    """
    n = sys.size

    if sys.size > 2:

        add_anc = QuantumRegister(n-2, 'add_anc')
        qc = QuantumCircuit(anc, add_anc, sys)
        qc.cx(control_qubit=anc[0], target_qubit=sys[0])
        qc.ccx(anc[0], sys[0], sys[1])  # control, control, target
        qc.ccx(anc[0], sys[0], add_anc[0])

        for i in range(add_anc.size - 1):
            qc.ccx(add_anc[i], sys[i+1], add_anc[i+1])
            qc.cx(control_qubit=add_anc[i+1], target_qubit=sys[i+2])
        
        qc.ccx(add_anc[-1], sys[-2], sys[-1])

        for j in range(add_anc.size - 1):
            qc.ccx(sys[-(j+3)], add_anc[-(j+2)], add_anc[-(j+1)])

        qc.ccx(anc[0], sys[0], add_anc[0])

    else:

        qc = QuantumCircuit(anc, sys)
        qc.cx(control_qubit=anc[0], target_qubit=sys[0])
        if sys.size == 2:
            qc.ccx(anc[0], sys[0], sys[1])  # control, control, target

    # TODO: Should U_tilde^\dagger be here?

    return qc


def circuit_adder_overlap(
    U_var: QuantumCircuit,
    U_tilde: QuantumCircuit,
    inverse:bool = False,
) -> QuantumCircuit:
    """
    
    """
    n = U_var.num_qubits

    anc = QuantumRegister(1, 'anc')
    sys = QuantumRegister(n, 'sys')

    if n > 2:
        add_anc = QuantumRegister(n-2, 'add_anc')
        qc = QuantumCircuit(anc, add_anc, sys)
    else:
        qc = QuantumCircuit(anc, sys)

    # Hadamard on ancilla
    qc.h(anc[0])

    # Controlled U_var
    U_gate = U_var.to_gate(label="U_var").control(1)
    qc.append(U_gate, [anc[0]] + list(sys))

    # QNPU (adder block)
    adder_block = build_adder_block(anc, sys)
    if inverse:
        adder_block = adder_block.inverse()
    qc.compose(adder_block, inplace=True)

    # Controlled U_tilde^\dagger
    U_tilde_gate = U_tilde.to_gate(label="U_tilde")
    U_tilde_dag_gate = U_tilde_gate.inverse()
    U_tilde_dag_gate.label = "U_tilde_dag"
    qc.append(U_tilde_dag_gate.control(1), [anc[0]] + list(sys))

    # Final Hadamard on ancilla
    qc.h(anc[0])

    return qc


def circuit_adder_overlap_2d(
    U_var: QuantumCircuit,
    U_tilde: QuantumCircuit,
    inverse:bool = False,
) -> tuple[QuantumCircuit, QuantumCircuit]:
    """
    
    """
    n = U_var.num_qubits

    anc = QuantumRegister(1, 'anc')
    sys_x = QuantumRegister(n//2, 'sys_x')
    sys_y = QuantumRegister(n//2, 'sys_y')

    if (n//2) > 2:
        add_anc = QuantumRegister((n//2)-2, 'add_anc')
        qc_x = QuantumCircuit(anc, add_anc, sys_x, sys_y)
    else:
        qc_x = QuantumCircuit(anc, sys_x, sys_y)

    # Hadamard on ancilla
    qc_x.h(anc[0])

    # Controlled U_var
    U_gate = U_var.to_gate(label="U_var").control(1)
    qc_x.append(U_gate, [anc[0]] + list(sys_x) + list(sys_y))

    # QNPU (adder block)
    qc_y = qc_x.copy() 
    adder_block_x = build_adder_block(anc, sys_x)
    adder_block_y = build_adder_block(anc, sys_y)
    if inverse:
        adder_block_x = adder_block_x.inverse()
        adder_block_y = adder_block_y.inverse()
    # Explicitly map adder blocks to the correct registers
    qc_x.compose(adder_block_x, qubits=[*anc, *sys_x], inplace=True)
    qc_y.compose(adder_block_y, qubits=[*anc, *sys_y], inplace=True)

    # Controlled U_tilde^\dagger
    U_tilde_gate = U_tilde.to_gate(label="U_tilde")
    U_tilde_dag_gate = U_tilde_gate.inverse()
    U_tilde_dag_gate.label = "U_tilde_dag"

    qc_x.append(U_tilde_dag_gate.control(1), [anc[0]] + list(sys_x) + list(sys_y))
    qc_y.append(U_tilde_dag_gate.control(1), [anc[0]] + list(sys_x) + list(sys_y))

    # Final Hadamard on ancilla
    qc_x.h(anc[0])
    qc_y.h(anc[0])

    return qc_x, qc_y

# TODO: molte cose da capire ancora
def circuit_diag_overlap(
    U_var: QuantumCircuit,
    U_tilde: QuantumCircuit, 
    inverse:bool = False
) -> QuantumCircuit:
    """
    Hadamard-test for w_D = Re <ψ~|D†_{ψ~}|ψ>, equivalent to Re{<ψ~|D_{ψ~}|ψ>} as D_f is
    a diagonal matrix with f_k values on the diagonal (real values). This is like 
    Fig. S3(c) but without the adder A.
    """
    n = U_var.num_qubits
    anc = QuantumRegister(1, 'anc')
    sys = QuantumRegister(n, 'sys')
    diag = QuantumRegister(n, 'diag')
    qc = QuantumCircuit(anc, sys, diag)

    qc.h(anc[0])

    # Controlled U_var
    qc.append(U_var.to_gate(label="U_var").control(1), [anc[0]] + list(sys))


    # Controlled Dψ~† (encoding of ψ~ amplitudes)
    diag_block = QuantumCircuit(anc, sys, diag)
    
    for k in range(n):
        diag_block.ccx(anc[0], sys[k], diag[k])

    if inverse:
        diag_block = diag_block.inverse()

    qc.compose(diag_block, inplace=True)

    U_tilde_gate = U_tilde.to_gate(label="U_tilde")
    U_tilde_dag_gate = U_tilde_gate.inverse()
    U_tilde_dag_gate.label = "U_tilde_dag"   
    qc.append(U_tilde_dag_gate.control(1), [anc[0]] + list(diag))

    # TODO: IS IT SUPPOSED TO BE HERE OR NOT?
    qc.append(U_tilde_dag_gate.control(1), [anc[0]] + list(sys))

    qc.h(anc[0])
    return qc


# TODO: is this Re{<ψ~|A D†_{ψ~}|ψ>} equivalent to Re{<ψ~|D_{ψ~}A|ψ>} correct?
def circuit_nonlinear_overlap(U_var: QuantumCircuit, U_tilde: QuantumCircuit) -> QuantumCircuit:
    """
    Fig. S3(c): Hadamard-test for <ψ~|A D†_{ψ~}|ψ>. 
    Re{<ψ~|A D†_{ψ~}|ψ>} equivalent to Re{<ψ~|D_{ψ~}A|ψ>}
    """
    n = U_var.num_qubits
    anc = QuantumRegister(1, 'anc')
    sys = QuantumRegister(n, 'sys')
    qnpu = QuantumRegister(2, 'qnpu')  # TODO: change name to adder
    diag = QuantumRegister(4, 'diag')
    qc = QuantumCircuit(anc, qnpu, sys, diag)

    # Hadamard on ancilla
    qc.h(anc[0])

    # Controlled U_var
    U_gate = U_var.to_gate(label="U_var").control(1)
    qc.append(U_gate, [anc[0]] + list(sys))

    # Controlled Dψ~† (diagonal encoding of amplitudes of ψ~)
    qc.ccx(sys[0], anc[0], diag[0])
    qc.ccx(sys[1], anc[0], diag[1])
    qc.ccx(sys[2], anc[0], diag[2])
    qc.ccx(sys[3], anc[0], diag[3])

    # Controlled U_tilde^\dagger
    U_tilde_gate = U_tilde.to_gate(label="U_tilde")
    U_tilde_dag_gate = U_tilde_gate.inverse()
    U_tilde_dag_gate.label = "U_tilde_dag"
    qc.append(U_tilde_dag_gate.control(1), [anc[0]] + list(diag))

    # Controlled Adder A
    qc.cx(control_qubit=anc[0], target_qubit=sys[0])
    qc.ccx(sys[0], anc[0], sys[1])
    qc.ccx(sys[0], anc[0], qnpu[0])
    qc.ccx(sys[1], qnpu[0], qnpu[1])
    qc.cx(control_qubit=qnpu[1], target_qubit=sys[2])
    qc.ccx(sys[2], qnpu[1], sys[3])
    qc.ccx(sys[1], qnpu[0], qnpu[1])
    qc.ccx(sys[0], anc[0], qnpu[0])

    # Controlled U_tilde^\dagger
    # U_tilde_gate = U_tilde.to_gate(label="U_tilde")
    # U_tilde_dag_gate = U_tilde_gate.inverse()
    # U_tilde_dag_gate.label = "U_tilde_dag"
    qc.append(U_tilde_dag_gate.control(1), [anc[0]] + list(sys))

    # Final Hadamard
    qc.h(anc[0])

    return qc

# TODO: is this Re{<ψ~|A D†_{ψ~}|ψ>} equivalent to Re{<ψ~|D_{ψ~}A|ψ>} correct?
def circuit_nonlinear_overlap_2d(
    U_var: QuantumCircuit, 
    U_tilde: QuantumCircuit,
    inverse:bool = False,
) -> tuple[QuantumCircuit, QuantumCircuit]:
    """
    Fig. S3(c): Hadamard-test for <ψ~|A D†_{ψ~}|ψ>. 
    Re{<ψ~|A D†_{ψ~}|ψ>}, equivalent to Re{<ψ~|D_{ψ~}A|ψ>}?
    """
    n = U_var.num_qubits

    anc = QuantumRegister(1, 'anc')
    sys_x = QuantumRegister(n//2, 'sys_x')
    sys_y = QuantumRegister(n//2, 'sys_y')
    diag = QuantumRegister(n, 'diag')

    if (n//2) > 2:
        add_anc = QuantumRegister((n//2)-2, 'add_anc')
        qc_x = QuantumCircuit(anc, add_anc, sys_x, sys_y, diag)
    else:
        qc_x = QuantumCircuit(anc, sys_x, sys_y, diag)

    # Hadamard on ancilla
    qc_x.h(anc[0])

    # Controlled U_var
    U_gate = U_var.to_gate(label="U_var").control(1)
    qc_x.append(U_gate, [anc[0]] + list(sys_x) + list(sys_y))

    # # Controlled Dψ~† (encoding of ψ~ amplitudes)
    # diag_block = QuantumCircuit(anc, sys_x, sys_y, diag)
    
    for k in range(n):
        if k < n//2:
            qc_x.ccx(anc[0], sys_x[k], diag[k])
        else:
            qc_x.ccx(anc[0], sys_y[k - n//2], diag[k])

    # QNPU (adder block)
    adder_block_x = build_adder_block(anc, sys_x)
    # diag_adder_block_x = diag_block.append(adder_block_x, qubits=[*anc, *sys_x], inplace=True)
    adder_block_y = build_adder_block(anc, sys_y)
    # diag_adder_block_y = diag_block.append(adder_block_y, qubits=[*anc, *sys_y], inplace=True)
    if inverse:
        raise NotImplementedError("Inverse not implemented yet for 2D adder blocks")
    
    # Controlled U_tilde^\dagger diag
    U_tilde_gate = U_tilde.to_gate(label="U_tilde")
    U_tilde_dag_gate = U_tilde_gate.inverse()
    U_tilde_dag_gate.label = "U_tilde_dag"

    qc_y = qc_x.copy()
    qc_x.append(U_tilde_dag_gate.control(1), [anc[0]] + list(diag))
    qc_y.append(U_tilde_dag_gate.control(1), [anc[0]] + list(diag))

    # Explicitly map adder blocks to the correct registers 
    qc_x.compose(adder_block_x, qubits=[*anc, *sys_x], inplace=True)
    qc_y.compose(adder_block_y, qubits=[*anc, *sys_y], inplace=True)

    # Controlled U_tilde^\dagger sys
    qc_x.append(U_tilde_dag_gate.control(1), [anc[0]] + list(sys_x) + list(sys_y))
    qc_y.append(U_tilde_dag_gate.control(1), [anc[0]] + list(sys_x) + list(sys_y))

    # Final Hadamard on ancilla
    qc_x.h(anc[0])
    qc_y.h(anc[0])

    return qc_x, qc_y

    
# TODO: refine base class
class BasePDE(ABC):
    """Base class for PDE loss functions."""
    
    @property
    def name(self) -> str:
        return self.__class__._name

    def __init__(self) -> None:
        pass

    def ancilla_z_exp(
        self,
        circ: QuantumCircuit, 
        shots: Optional[float]=None
    ) -> float:
        """
        Compute <Z> on the ancilla qubit (anc[0]) of a Hadamard-test circuit.
        circ: QuantumCircuit with ancilla register named 'anc'
        shots: if None -> exact statevector expectation
            if int  -> run qasm simulator with that many shots
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
                anc_bit = (idx >> n-1) & 1  # TODO: check if correct
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
            # ancilla is most significant qubit => look at first char of key
            N0, N1 = 0, 0
            # TODO: not sure it's 0 or n-1
            for bitstring, count in counts.items():
                if bitstring[0] == '0':
                    N0 += count
                else:
                    N1 += count
            return (N0 - N1) / shots

    @abstractmethod
    def cost(self, lambdas) -> float:
        pass


class Burgers(BasePDE):
    """
    Implements the Burgers-Euler step cost (Eq. S4):

      C(λ0, λ) = |λ0|^2
                 - 2 Re{ λ0 · (tilde_λ0)^* <0| Ũ^† (1 + τ O) U(λ) |0> } + const

    with  O = ν Δ - tilde_λ0 · diag(tilde_f) ∇, tilde_f = tilde_λ0 · tilde_ψ, and Ũ|0> = tilde_ψ.

    Notes:
    - This computes C without the constant (which does not affect minimization).
    - Uses a single-parameter real ansatz: Ry(λ) only + CNOT chain.
    """

    # TODO: gather all the relevant params in a placeholder class like 
    def __init__(
        self, 
        lambda0: float,
        lambdas: np.ndarray | list,
        nu: float, 
        tau:float,
        n_qubits: int,
        depth: int, 
    ) -> None:
        """
        Args:
            lambda0: Current step scalar λ0
            lam: Current step single variational parameter λ (real)
        """
        self.lambda0 = lambda0
        self.lambdas = lambdas
        self.nu = nu
        self.tau = tau
        self.n_qubits = n_qubits
        self.depth = depth


    def cost(self, lambdas):
        """
        Compute cost C(lambda0, lambda) up to the additive const term, by running the Hadamard-test circuits.

        Returns:
            dict with measured primitives and assembled cost_up_to_const
                {
                'w0':..., 'wA':..., 'wAinv':..., 'wD':..., 'wDA':...,
                'LapVal':..., 'NonlinVal':..., 's':..., 'cost_up_to_const':...
                }
        """
        lambda0_new = float(lambdas[0])
        lambdas_new = np.array(lambdas[1:], copy=True)  # TODO: needed

        # 0) build current variational circuit U_var from ansatz (SingleParameterAnsatz)
        ansatz = HEAnsatz(n_qubits=self.n_qubits, depth=self.depth)
        U_tilde_circ = ansatz.qc(self.lambdas)
        U_var = ansatz.qc(lambdas_new)

        # 1) build the 5 hadamard-test circuits (each returns ancilla expectation)
        qc_w0 = circuit_overlap(U_var, U_tilde_circ)                       # w0
        qc_wA = circuit_adder_overlap(U_var, U_tilde_circ, inverse=False) # wA
        qc_wAinv = circuit_adder_overlap(U_var, U_tilde_circ, inverse=True)# wA^{-1}
        qc_wD = circuit_diag_overlap(U_var, U_tilde_circ, inverse=True)   # wD (diag only)
        qc_wDA = circuit_nonlinear_overlap(U_var, U_tilde_circ)            # wDA (D then A / or equivalent)

        # 2) run circuits (use exact statevector for now)
        # Note: ancilla_z_exp assumes ancilla is qubit index 0 (first register)
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

        self.lambda0 = lambda0_new
        self.lambdas = lambdas_new

        return cost


class Burgers2D(Burgers):
    """
    """

    def cost(self, lambdas):
        """
        
        """
        lambda0_new = float(lambdas[0])
        lambdas_new = np.array(lambdas[1:], copy=True)  # TODO: needed

        # 0) build current variational circuit U_var from ansatz (SingleParameterAnsatz)
        ansatz = HEAnsatz(n_qubits=self.n_qubits, depth=self.depth)
        U_tilde_circ = ansatz.qc(self.lambdas)
        U_var = ansatz.qc(lambdas_new)

        # 1) build the hadamard-test circuits
        qc_w0 = circuit_overlap(U_var, U_tilde_circ)                                    # w0
        qc_wA_x, qc_wA_y = circuit_adder_overlap_2d(U_var, U_tilde_circ, inverse=False) # wA
        qc_wAinv_x, qc_wAinv_y = circuit_adder_overlap_2d(U_var, U_tilde_circ, inverse=True)  # wA^{-1}
        qc_wD = circuit_diag_overlap(U_var, U_tilde_circ, inverse=True)                 # wD (diag only)
        qc_wDA_x, qc_wDA_y = circuit_nonlinear_overlap_2d(U_var, U_tilde_circ)          # wDA (D then A / or equivalent)

        # 2) run circuits (use exact statevector for now)
        # Note: ancilla_z_exp assumes ancilla is qubit index 0 (first register)
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

        self.lambda0 = lambda0_new
        self.lambdas = lambdas_new

        return cost


class CostFunction:
    """
    Implements the Burgers-Euler step cost (Eq. S4):

      C(λ0, λ) = |λ0|^2
                 - 2 Re{ λ0 · (tilde_λ0)^* <0| Ũ^† (1 + τ O) U(λ) |0> } + const

    with  O = ν Δ - tilde_λ0 · diag(tilde_f) ∇, tilde_f = tilde_λ0 · tilde_ψ, and Ũ|0> = tilde_ψ.

    Notes:
    - This computes C without the constant (which does not affect minimization).
    - Uses a single-parameter real ansatz: Ry(λ) only + CNOT chain.
    """

    # TODO: gather all the relevant params in a placeholder class like 
    def __init__(
        self, 
        lambda0: float,
        lambda1: float,
        nu: float, 
        tau:float,
        n_qubits: int,
        depth: int, 
    ) -> None:
        """
        Args:
            lambda0: Current step scalar λ0
            lam: Current step single variational parameter λ (real)
        """
        self.lambda0 = float(lambda0)
        self.lambda1 = float(lambda1)
        self.nu = nu
        self.tau = tau
        self.n_qubits = n_qubits
        self.depth = depth

    
    def ancilla_z_exp(self, circ: QuantumCircuit, shots: Optional[float]=None) -> float:
        """
        Compute <Z> on the ancilla qubit (anc[0]) of a Hadamard-test circuit.
        circ: QuantumCircuit with ancilla register named 'anc'
        shots: if None -> exact statevector expectation
            if int  -> run qasm simulator with that many shots
        """
        if shots is None:
            # Exact expectation from statevector
            backend = Aer.get_backend("statevector_simulator")
            t_qc = transpile(circ, backend)
            sv = backend.run(t_qc).result().get_statevector(t_qc)
            n = circ.num_qubits
            probs = np.abs(sv) ** 2
            p0, p1 = 0.0, 0.0

            # TODO: simplify here, acnilla is always either n-1 or 0
            # locate ancilla qubit position
            # anc = [q for q in circ.qubits if q._register.name == 'anc'][0]
            # anc_pos = t_qc.find_bit(anc).index
            # if self.first: # TODO: remove
            #     print("\n\nANCILLA POSITION:", anc_pos, "\n\n")  # it prints out 0
            #     self.first = False

            for idx, p in enumerate(probs):
                anc_bit = (idx >> n-1) & 1  # TODO: check if correct
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
            # ancilla is most significant qubit => look at first char of key
            N0, N1 = 0, 0
            # TODO: not sure it's 0 or n-1
            for bitstring, count in counts.items():
                if bitstring[0] == '0':
                    N0 += count
                else:
                    N1 += count
            return (N0 - N1) / shots


    def cost(self, lambdas):
        """
        Compute cost C(lambda0, lambda) up to the additive const term, by running the Hadamard-test circuits.

        Returns:
            dict with measured primitives and assembled cost_up_to_const
                {
                'w0':..., 'wA':..., 'wAinv':..., 'wD':..., 'wDA':...,
                'LapVal':..., 'NonlinVal':..., 's':..., 'cost_up_to_const':...
                }
        """
        # 0) build current variational circuit U_var from ansatz (SingleParameterAnsatz)
        ansatz = SingleParameterAnsatz(n_qubits=self.n_qubits, depth=self.depth)
        U_tilde_circ = ansatz.qc(self.lambda1)
        U_var = ansatz.qc(lambdas[1])

        # 1) build the 5 hadamard-test circuits (each returns ancilla expectation)
        qc_w0 = circuit_overlap(U_var, U_tilde_circ)                       # w0
        qc_wA = circuit_adder_overlap(U_var, U_tilde_circ, inverse=False) # wA
        qc_wAinv = circuit_adder_overlap(U_var, U_tilde_circ, inverse=True)# wA^{-1}
        qc_wD = circuit_diag_overlap(U_var, U_tilde_circ, inverse=False)   # wD (diag only)
        qc_wDA = circuit_nonlinear_overlap(U_var, U_tilde_circ)            # wDA (D then A / or equivalent)

        # 2) run circuits (use exact statevector for now)
        # Note: ancilla_z_exp assumes ancilla is qubit index 0 (first register)
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
        cost = (lambdas[0] ** 2) - 2.0 * np.real(lambdas[0] * np.conj(self.lambda0)) * s

        self.lambda0 = lambdas[0]
        self.lambda1 = lambdas[1]

        return cost