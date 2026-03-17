from typing import Callable, Sequence

import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import numpyro.distributions.transforms as transforms

from pmrf.model import Model
from pmrf.frequency import Frequency
from pmrf.transforms import HypercubeTransform
from pmrf.goal import Goal
from pmrf.optimize.problem import OptimizeProblem
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.backends.optimistix import OptimistixOptimizer

class AbstractOptimizer(eqx.Module):
    """
    The unified interface for all ParamRF optimization backends.
    Inherits from eqx.Module to ensure it can be safely passed around in JAX transforms if needed.
    """
    def solve(self, problem: OptimizeProblem) -> OptimizeResult:
        """
        Executes the optimization routine on the given problem.
        """
        raise NotImplementedError("Each optimizer backend must implement the solve method.")
    
def optimize_model(
    model: Model,
    frequency: Frequency,
    cost: Callable[[Model, Frequency], jnp.ndarray] | Sequence[Goal],
    solver: AbstractOptimizer | optx.AbstractMinimiser = optx.BFGS(),
) -> OptimizeResult:
    if isinstance(solver, optx.AbstractMinimiser):
        solver = OptimistixOptimizer(solver)

    problem = OptimizeProblem(model, frequency, cost)
    results = solver.solve(problem)
    return results