from typing import TYPE_CHECKING

import jax.numpy as jnp

import equinox as eqx
import optimistix as optx

from pmrf.core import Model, Frequency, Evaluator, Problem, partition
from pmrf.optimize.result import OptimizeResult
from pmrf.network_collection import NetworkCollection

if TYPE_CHECKING:
    import skrf

def fit_data(
    model: Model,
    frequency: Frequency,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    solver: optx.AbstractMinimiser | None = None,
) -> OptimizeResult:
    return