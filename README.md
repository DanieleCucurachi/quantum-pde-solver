
# Quantum PDE Solver

A research-oriented Python package that implements a variational quantum algorithm for solving nonlinear PDEs using a forward Euler time-stepping scheme as proposed in [Lubasch et al., “Variational quantum algorithms for nonlinear problems” (PR A, 2019)](https://arxiv.org/pdf/1907.09032).

**See [`docs/improvements.md`](https://github.com/DanieleCucurachi/quantum-pde-solver/blob/main/docs/improvements.md) for possible improvements to this proof-of-principle solver.**

**See [`docs/method_description_and_validation.md`](https://github.com/DanieleCucurachi/quantum-pde-solver/blob/main/docs/method_description_and_validation.md) for a short description of the pipeline implementig the algorithm.**

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

### Creating an Environment
You can also set up the environment using the provided environment.yml file:
```bash
conda env create -f environment.yml
conda activate quantum-pde-solver
```
This will install all required dependencies in a new environment named quantum-pde-solver.

---

## Usage

Start with [`examples/tutorial.ipynb`](https://github.com/DanieleCucurachi/quantum-pde-solver/blob/main/examples/tutorial.ipynb) for a quick overview of what you can do with this repo.

### 1. Quantum PDE Solver

Solve the 2D diffusion equation with:

```bash
python examples/diffusion_2d.py --n_qubits 4 --depth 2 --tau 0.1 --D 0.1 --tmax 5.0 --sigma 0.15 --seed 42
```

All simulation parameters can be set via command-line arguments. The same command can be used for diffusion 1 and 2D, and Burgers 1 and 2D, just make sure to use the appropriate constants ($D$ for diffusion, $\nu$ for Burgers).

### 2. Circuit Blocks Creation

[`examples/circuit_blocks.ipynb`](https://github.com/DanieleCucurachi/quantum-pde-solver/blob/main/examples/circuit_blocks.ipynb) contains examples on how to construct the fundamental circuit blocks described in [Lubasch et al., “Variational quantum algorithms for nonlinear problems” (PR A, 2019)](https://arxiv.org/pdf/1907.09032).

### 3. Initial State Preparation

In [`examples/initial_state_prep.ipynb`](https://github.com/DanieleCucurachi/quantum-pde-solver/blob/main/examples/initial_state_prep.ipynb) a variational quantum circuit is optimized to prepare a Gaussian state (1D and 2D case are shown).

### 4. Classical Reference Solutions

[`examples/classical_solver.ipynb`](https://github.com/DanieleCucurachi/quantum-pde-solver/blob/main/examples/classical_solver.ipynb) contains a tutorial showing how to solve and visualize 1D/2D Burgers' and Diffusion equations using FiPy.

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
|   ├── tutorial.ipynb                # Tutorial basic functionalities of the repo
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
