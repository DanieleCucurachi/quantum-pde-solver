from __future__ import annotations


from qiskit import QuantumCircuit, QuantumRegister


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
            qc.ccx(anc[0], sys[0], sys[1])

    return qc


def circuit_adder_overlap_1d(
    U_var: QuantumCircuit,
    U_tilde: QuantumCircuit,
    inverse:bool = False,
) -> QuantumCircuit:
    """
    Construct the 1D Hadamard test circuit with an adder (QNPU) block.

    This circuit implements a Hadamard test for Re⟨ψ̃|A|ψ⟩, where A is the adder block.
    The circuit applies controlled-U_var, the adder (or its inverse), controlled-U_tilde†,
    and Hadamard gates on the ancilla.

    Args:
        U_var (QuantumCircuit): Circuit preparing |ψ⟩.
        U_tilde (QuantumCircuit): Circuit preparing |ψ̃⟩.
        inverse (bool): If True, use the inverse adder block.

    Returns:
        QuantumCircuit: The full Hadamard test circuit with adder.
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
    Construct the 2D Hadamard test circuit with an adder (QNPU) block.

    This circuit implements a Hadamard test for Re⟨ψ̃|A|ψ⟩, where A is the adder block.
    The circuit applies controlled-U_var, the adder (or its inverse), controlled-U_tilde†,
    and Hadamard gates on the ancilla.

    Args:
        U_var (QuantumCircuit): Circuit preparing |ψ⟩.
        U_tilde (QuantumCircuit): Circuit preparing |ψ̃⟩.
        inverse (bool): If True, use the inverse adder block.

    Returns:
        QuantumCircuit: The full Hadamard test circuit with adder.
    """
    n = U_var.num_qubits

    if n % 2 != 0:
        raise ValueError("Number of qubits must be even for 2D adder overlap circuit.")

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


def circuit_diag_overlap(
    U_var: QuantumCircuit,
    U_tilde: QuantumCircuit, 
    inverse:bool = False
) -> QuantumCircuit:
    """
    Construct a Hadamard test circuit for measuring Re⟨ψ̃|D†_{ψ̃}|ψ⟩, where D is a
    diagonal operator in the computational basis.
    This is equivalent to Fig. S3(c) in Lubasch et al., but without the adder block.

    Args:
        U_var (QuantumCircuit): Quantum circuit preparing |ψ⟩.
        U_tilde (QuantumCircuit): Quantum circuit preparing |ψ̃⟩.
        inverse (bool): If True, use the inverse of the diagonal block.

    Returns:
        QuantumCircuit: The Hadamard test circuit for the diagonal observable.
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

    # Controlled U_tilde^\dagger
    U_tilde_gate = U_tilde.to_gate(label="U_tilde")
    U_tilde_dag_gate = U_tilde_gate.inverse()
    U_tilde_dag_gate.label = "U_tilde_dag"   
    qc.append(U_tilde_dag_gate.control(1), [anc[0]] + list(diag))

    # TODO: check
    qc.append(U_tilde_dag_gate.control(1), [anc[0]] + list(sys))

    qc.h(anc[0])
    return qc


def circuit_nonlinear_overlap_1d(
    U_var: QuantumCircuit, 
    U_tilde: QuantumCircuit,
    inverse:bool = False,
) -> QuantumCircuit:
    """
    Construct a Hadamard test circuit for measuring Re⟨ψ̃|A D†_{ψ̃}|ψ⟩,
    where A is the adder block and D is a diagonal operator in the computational basis.
    This corresponds to Fig. S3(c) in Lubasch et al., with both adder and diagonal blocks.

    Args:
        U_var (QuantumCircuit): Quantum circuit preparing |ψ⟩.
        U_tilde (QuantumCircuit): Quantum circuit preparing |ψ̃⟩.

    Returns:
        QuantumCircuit: The Hadamard test circuit for the nonlinear observable.
    """
    n = U_var.num_qubits

    anc = QuantumRegister(1, 'anc')
    sys = QuantumRegister(n, 'sys')
    diag = QuantumRegister(n, 'diag')

    if n > 2:
        add_anc = QuantumRegister(n-2, 'add_anc')
        qc = QuantumCircuit(anc, add_anc, sys, diag)
    else:
        qc = QuantumCircuit(anc, sys, diag)

    # Hadamard on ancilla
    qc.h(anc[0])

    # Controlled U_var
    U_gate = U_var.to_gate(label="U_var").control(1)
    qc.append(U_gate, [anc[0]] + list(sys))

    # Controlled Dψ~† (diagonal encoding of amplitudes of ψ~)
    for k in range(n):
        qc.ccx(anc[0], sys[k], diag[k])

    # Controlled U_tilde^\dagger
    U_tilde_gate = U_tilde.to_gate(label="U_tilde")
    U_tilde_dag_gate = U_tilde_gate.inverse()
    U_tilde_dag_gate.label = "U_tilde_dag"
    qc.append(U_tilde_dag_gate.control(1), [anc[0]] + list(diag))

    # QNPU (adder block)
    adder_block = build_adder_block(anc, sys)
    # TODO: add option to invert diag block too?
    if inverse:
        adder_block = adder_block.inverse()
    qc.compose(adder_block, inplace=True)

    # Controlled U_tilde^\dagger
    qc.append(U_tilde_dag_gate.control(1), [anc[0]] + list(sys))

    # Final Hadamard
    qc.h(anc[0])

    return qc


def circuit_nonlinear_overlap_2d(
    U_var: QuantumCircuit, 
    U_tilde: QuantumCircuit,
    inverse:bool = False,
) -> tuple[QuantumCircuit, QuantumCircuit]:
    """
    Construct a Hadamard test circuit for measuring Re⟨ψ̃|A D†_{ψ̃}|ψ⟩,
    where A is the adder block and D is a diagonal operator in the computational basis.
    This corresponds to Fig. S3(c) in Lubasch et al., with both adder and diagonal blocks.

    Args:
        U_var (QuantumCircuit): Quantum circuit preparing |ψ⟩.
        U_tilde (QuantumCircuit): Quantum circuit preparing |ψ̃⟩.

    Returns:
        QuantumCircuit: The Hadamard test circuit for the nonlinear observable.
    """
    n = U_var.num_qubits

    if n % 2 != 0:
        raise ValueError("Number of qubits must be even for 2D nonlinear overlap circuit.")

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
    for k in range(n):
        if k < n//2:
            qc_x.ccx(anc[0], sys_x[k], diag[k])
        else:
            qc_x.ccx(anc[0], sys_y[k - n//2], diag[k])

    # QNPU (adder block)
    adder_block_x = build_adder_block(anc, sys_x)
    adder_block_y = build_adder_block(anc, sys_y)
    # TODO: add option to invert diag block too?
    if inverse:
        adder_block_x = adder_block_x.inverse()
        adder_block_y = adder_block_y.inverse()

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