from typing import Callable, Sequence

import jax.numpy as jnp
import equinox as eqx
import optimistix as optx

from pmrf.model import Model
from pmrf.frequency import Frequency
from pmrf.optimize.goal import Goal
from pmrf.optimize.problem import OptimizeProblem
from pmrf.optimize.result import OptimizeResult

class AbstractOptimizer(eqx.Module):
    """
    The unified interface for all ParamRF optimization backends.
    Inherits from eqx.Module to ensure it can be safely passed around in JAX transforms if needed.
    """
    def solve(self, problem: OptimizeProblem, show_progress: bool = False, **kwargs) -> OptimizeResult:
        """
        Executes the optimization routine on the given problem.
        """
        raise NotImplementedError("Each optimizer backend must implement the solve method.")
    
def optimize_model(
    model: Model,
    frequency: Frequency,
    cost: Callable[[Model, Frequency], jnp.ndarray] | Sequence[Goal],
    solver: AbstractOptimizer | optx.AbstractMinimiser,
    **kwargs,
) -> OptimizeResult:
    if isinstance(solver, optx.AbstractMinimiser):
        from pmrf.optimize.backends.optimistix import OptimistixOptimizer
        solver = OptimistixOptimizer(solver)

    problem = OptimizeProblem(model, frequency, cost)

    results = solver.solve(problem, **kwargs)

    return results