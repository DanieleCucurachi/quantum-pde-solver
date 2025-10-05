
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

### Creating an Environment
You can also set up the environment using the provided environment.yml file:
```bash
conda env create -f [environment.yml](http://_vscodecontentref_/0)
conda activate quantum-pde-solver
```
This will install all required dependencies in a new environment named quantum-pde-solver.

---

## Usage

### 1. Quantum PDE Solver

Solve the 2D Burgers' equation with:

```bash
python examples/burgers_2d.py --n_qubits 4 --depth 2 --tau 0.1 --nu 0.1 --tmax 5.0 --sigma 0.15 --seed 42
```

All simulation parameters can be set via command-line arguments. The same command can be used for diffusion 1 and 2D, and Burgers 1D.

### 2. Circuit Blocks Creation

See [`examples/circuit_blocks.ipynb`](https://github.com/DanieleCucurachi/quantum-pde-solver/blob/main/examples/circuit_blocks.ipynb) tutorial for step-by-step examples on how to construct the fundamental quantum circuit blocks described in [Lubasch et al., “Variational quantum algorithms for nonlinear problems” (PR A, 2019)](https://arxiv.org/pdf/1907.09032).

### 3. Initial State Preparation

Optimize a quantum circuit to prepare a target Gaussian state:

- See [`examples/initial_state_prep.ipynb`](https://github.com/DanieleCucurachi/quantum-pde-solver/blob/main/examples/initial_state_prep.ipynb) for 1D and 2D examples.

### 4. Classical Reference Solutions

Jupyter notebooks in `examples/` demonstrate how to use the classical solvers to compare with quantum results:

- [`examples/classical_solver.ipynb`](https://github.com/DanieleCucurachi/quantum-pde-solver/blob/main/examples/classical_solver.ipynb) — Solve and visualize 1D/2D Burgers' and Diffusion equations using FiPy.

---

## Repository Structure

```
quantum-pde-solver/
│
├── src/
│   ├── ansatz.py             # Quantum circuit ansatz definitions
│   ├── circuit.py            # Quantum circuit construction (Hadamard tests, etc.)
│   ├── pdes.py               # PDE cost function classes (Burgers1D, Burgers2D, etc.)
│   ├── plot.py               # Plotting utilities
│   ├── ref_solutions.py      # Classical PDE solvers (FiPy)
│   ├── time_evo.py           # Time evolution and training routines
│   ├── utils.py              # Utilities (state prep, fidelity, etc.)
│   └── ...
│
├── examples/
│   ├── burgers_1d.py                 # Script for 1D Burgers simulation
│   ├── burgers_2d.py                 # Script for 2D Burgers simulation
|   ├── diffusion_1d.py               # Script for 1D Diffusion simulation
│   ├── diffusion_2d.py               # Script for 2D Diffusion simulation
│   ├── circuit_blocks.ipynb          # Notebook: quantum circuit blocks demos
│   ├── classical_solver.ipynb        # Classical PDE solver notebook
│   ├── initial_state_prep.ipynb      # Initial state preparation notebook
│   ├── images/                       # Images for notebooks and documentation
│   │   └── fig_s3_lubasch.png
│   └── exp_results/                  # Output directory for results and plots
│
├── setup.py
├── README.md
├── LICENSE
├── environment.yml
└── ...
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

```
MIT License

Copyright (c) 2025 Daniele Cucurachi

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