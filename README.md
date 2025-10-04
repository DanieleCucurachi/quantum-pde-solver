
# Quantum PDE Solver

A research-oriented Python package that implements a variational quantum algorithm for solving nonlinear PDEs using a forward Euler time-stepping scheme as proposed in [Lubasch et al., “Variational quantum algorithms for nonlinear problems” (PR A, 2019)](https://arxiv.org/pdf/1907.09032).


## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/quantum-pde-solver.git
cd quantum-pde-solver
pip install -e .
```

**Requirements:**  
- Python 3.8+
- numpy, scipy, matplotlib, pandas
- qiskit, qiskit-aer
- fipy
- typing_extensions

---

## Usage

### 1. Quantum PDE Solver

Solve the 2D Burgers' equation with:

```bash
python examples/burgers_2d.py --n_qubits 4 --depth 2 --tau 0.1 --nu 0.1 --tmax 5.0 --sigma 0.15 --seed 42
```

All simulation parameters can be set via command-line arguments.

### 2. Classical Reference Solutions

Jupyter notebooks in `examples/` demonstrate how to use the classical solvers to compare with quantum results:

- [`examples/classical_solver.ipynb`](https://github.com/DanieleCucurachi/quantum-pde-solver/blob/main/examples/classical_solver.ipynb) — Solve and visualize 1D/2D Burgers' and Diffusion equations using FiPy.

### 3. Initial State Preparation

Optimize a quantum circuit to prepare a target Gaussian state:

- See [`examples/initial_state_prep.ipynb`](https://github.com/DanieleCucurachi/quantum-pde-solver/blob/main/examples/initial_state_prep.ipynb) for 1D and 2D examples.

---

## Repository Structure

```
quantum-pde-solver/
│
├── src/
│   ├── ansatz.py         # Quantum circuit ansatz definitions
│   ├── circuit.py        # Quantum circuit construction (Hadamard tests, etc.)
│   ├── pdes.py           # PDE cost function classes (Burgers1D, Burgers2D, etc.)
│   ├── plot.py           # Plotting utilities
│   ├── ref_solutions.py  # Classical PDE solvers (FiPy)
│   ├── time_evo.py       # Time evolution and training routines
│   ├── utils.py          # Utilities (state prep, fidelity, etc.)
│   └── ...
│
├── examples/
|   ├── burgers_1d.py                 # Script for 1D quantum Burgers simulation
│   ├── burgers_2d.py                 # Script for 2D quantum Burgers simulation
│   ├── classical_solver.ipynb        # Classical PDE solver notebook
│   ├── initial_state_prep.ipynb      # Initial state preparation notebook
│   └── exp_results/                  # Output directory for results and plots
│
├── setup.py
├── README.md
└── LICENSE
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

```
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```