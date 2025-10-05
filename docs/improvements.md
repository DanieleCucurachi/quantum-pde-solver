# Future Improvements

## Algorithmic Improvements

### 1. Adaptive Time Stepping

Adaptive time stepping consists in changing the size of the time step $\tau$ during the time evolution to balance stability/accuracy and efficiency. 

Let us start by picking a tolerance $C_{\rm tol}$. After optimizing for a timestep $\tau_n$, compute the residual cost:

$$ C_{\rm res} = \bigl\|\lambda_0|\psi(\lambda)\rangle - (1 + \tau_n\,\hat O)\,\tilde\lambda_0|\tilde\psi\rangle\bigr\|^2.$$

Then:

- If $C_{\rm res} > C_{\rm tol}$, the step is too aggressive ⇒ reduce:
  $$
  \tau_{n+1} = \max\bigl(\tau_n \cdot \alpha_{\rm dec},\; \tau_{\min}\bigr),
  $$
  where $\alpha_{\rm dec} < 1$ (e.g. 0.5) and $\tau_{\min}$ is a lower bound.

- Else (step acceptable), accept and optionally increase:
  $$
  \tau_{n+1} = \min\bigl(\tau_n \cdot \alpha_{\rm inc},\; \tau_{\max}\bigr),
  $$
  with $\alpha_{\rm inc} > 1$ (e.g. 1.2) and upper bound $\tau_{\max}$.

Optionally, require $C_{\rm res} \le (1 + \varepsilon)\,C_{\rm prev}$ (with small $\varepsilon$) to guard against drift (as in drifting away from stability). 

This schedule lets you shrink τ when cost signals instability, and grow τ in smooth regimes, adapting step sizes to maintain both accuracy and performance.



## Engineering Improvements

### 1. Optimize the Simulation Pipeline

There are several minor improvements

### 2. Better Optimizers & Differentiable Backends  

One major bottleneck in the variational time-stepping pipeline is the optimization of cost functions via derivative-free methods like COBYLA. You can gain speed and stability by switching to gradient-based or hybrid schemes, especially when the cost is differentiable (which it is in this case, as it's the output of a quantum circuit). 

For example, integrating libraries such as pyqtorch or JAX allows automatic differentiation, JIT compilation, and hardware acceleration (e.g. GPU), which would lower the per-step optimization overhead.

Another possible improvement could come from custom optimizers which work well for optimizing parametrized quantum circuits, such as quantum natural gradient.

#### Parallel / Distributed Circuit Evaluation  

Another practical improvement is to parallelize the evaluation of multiple quantum circuits across multiple CPU cores (if running on a laptop) or nodes (if running on HPC cluster). For instance with `dask` you can issue each circuit required for computing the cost (e.g. for $A_x$, $A_y$, $D$, etc.) as an independent task. By distributing circuit simulation and measurement, the wall-clock time per time step can drop significantly.

#### Batch & Vectorized Evaluations  
When your variational ansatz supports vectorization (e.g. through JAX’s vmap or PyTorch’s batch processing), you can evaluate multiple parameter candidates (or measurement circuits) in a single pass. This is useful in population-based or small exploration steps, where you evaluate cost/gradients for multiple $\lambda$ values in parallel. Some quantum-ML frameworks support batching of circuits or parallel evaluation primitives. This reduces overhead in cases where you want to explore nearby parameter sets or compute approximate gradients in batch.

#### Warm Starts and Step Reuse  
Within a time-stepping loop, many circuits and ansatz structures remain largely the same from one time step to the next. You can reuse compiled/transpiled gate sequences, initial parameter guesses, and even previously computed expectation values as starting points for the next time step’s optimizer. This “warm start” strategy reduces redundant work in compilation or circuit construction and helps the optimizer converge faster.

#### Hybrid Accelerator Use & Mixed Precision  
Leverage hardware accelerators (GPUs, TPUs) using frameworks like JAX to offload linear algebra, continuous optimization, and expectation-value arithmetic. Additionally, use lower-precision arithmetic (e.g. float32 or bfloat16) in parts of the computation where numerical stability allows, to increase throughput. Combined with auto-differentiable backends, this can drastically reduce CPU load and speed up the end-to-end loop.

---

