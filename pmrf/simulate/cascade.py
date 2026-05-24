"""pmrf/simulate/cascade.py"""

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
from typing import Sequence

from pmrf.frequency import Frequency
from pmrf.models.base import Model
from pmrf.simulate.base import AbstractCascader, AbstractScatteringCascader, AbstractTransferCascader
from pmrf.simulate.result import SimulateResult

def cascade(
    models: Sequence[Model],
    frequency: Frequency,
    solver: AbstractCascader,
    z0: ArrayLike = 50.0,
) -> SimulateResult:
    """
    Cascades a chain of 2N-port components in series.
    
    Args:
        models: An ordered sequence of Models to cascade (Left to Right).
        frequency: The frequency sweep over which to characterize the network.
        solver: An instance of a cascader algorithm (e.g., Redheffer or TransferMatmul).
        z0: The characteristic impedance for parameter evaluation.
        
    Returns:
        SimulateResult: A structured result containing the final cascaded matrices.
    """
    if not models:
        raise ValueError("Cannot cascade an empty list of models.")

    if isinstance(solver, AbstractScatteringCascader):
        
        # Directly stack the S-matrices from the model list
        # Transpose to (F, N_models, m, m)
        batched_S = jnp.stack([m.s(frequency, z0=z0) for m in models]).transpose(1, 0, 2, 3)
        
        n_models = len(models)
        m_ports = models[0].nports
        batched_z0 = jnp.broadcast_to(jnp.asarray(z0), (n_models, m_ports))
        
        if not jnp.isscalar(z0):
            raise ValueError("Cascade currently only accepts scalar characteristic impedances")
            
        vmapped_solver = jax.vmap(solver.run, in_axes=(0, None))
        solution = vmapped_solver(batched_S, batched_z0)
        
        return SimulateResult(solution=solution, z0=z0)
        
    elif isinstance(solver, AbstractTransferCascader):
        
        # Directly stack the A-matrices from the model list
        batched_A = jnp.stack([m.a(frequency) for m in models]).transpose(1, 0, 2, 3)
        
        vmapped_solver = jax.vmap(solver.run, in_axes=(0,))
        solution = vmapped_solver(batched_A)
        
        return SimulateResult(solution=solution, z0=z0)
        
    else:
        raise TypeError(f"Unrecognized cascader type: {type(solver)}")