from __future__ import annotations

from .ansatz import (
    HEAnsatz,
    SingleParameterAnsatz,
)

from .pdes import(
    Burgers1D,
    Burgers2D,
    Diffusion1D,
    Diffusion2D,
)

from .utils import (
    fidelity,
    set_seeds,
    gaussian_state,
    amplitude_encoding_2d,
)

from .ref_solutions import (
    DiffusionFiPySolver,
    BurgersFiPySolver,
)

from .time_evo import (
    prepare_initial_state,
    run_time_evolution,
)   