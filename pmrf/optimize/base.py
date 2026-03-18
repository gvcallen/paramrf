from typing import Callable, Any

import jax.numpy as jnp
import equinox as eqx

from pmrf.core import Model
from pmrf.optimize.result import OptimizeResult

class AbstractMinimizer(eqx.Module):
    """
    The unified interface for all ParamRF optimization backends.
    Inherits from eqx.Module to ensure it can be safely passed around in JAX transforms if needed.
    """
    def solve(
        self,
        model: Model,
        cost: Callable[[Model], jnp.ndarray]
    ) -> OptimizeResult:
        """
        Executes the optimization routine on the given problem.
        """
        raise NotImplementedError("Each optimizer backend must implement the solve method.")
    
def minimize(
    model: Model,
    cost: Callable[[Model], jnp.ndarray],
    args: Any,
    solver: AbstractMinimizer | None = None,
) -> OptimizeResult:

    if solver is None:
        import optimistix as optx
        solver = optx.LBFGS()

    if not isinstance(solver, AbstractMinimizer):
        import optimistix as optx
        if isinstance(solver, optx.AbstractMinimiser):
            from pmrf.optimize.backends.optimistix import OptimistixOptimizer
            solver = OptimistixOptimizer(solver)
        else:
            raise Exception(f"Unknown solver class: {type(solver)}")

    results = solver.solve(model, cost)

    return results