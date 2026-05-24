"""pmrf/simulate/terminate.py"""

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from pmrf.frequency import Frequency
from pmrf.models.base import Model
from pmrf.simulate.base import AbstractTerminator, AbstractScatteringTerminator, AbstractTransferTerminator
from pmrf.simulate.result import SimulateResult

def terminate(
    model_from: Model,
    model_into: Model,
    frequency: Frequency,
    solver: AbstractTerminator,
    z0: ArrayLike = 50.0,
) -> SimulateResult:
    """
    Terminates a source network into a load network.
    
    Args:
        model_from: The upstream/source Model.
        model_into: The downstream/load Model to terminate into.
        frequency: The frequency sweep over which to characterize the network.
        solver: An instance of a termination algorithm.
        z0: The characteristic impedance for parameter evaluation.
        
    Returns:
        SimulateResult: A structured result containing the terminated matrix.
    """
    if not jnp.isscalar(z0):
        raise ValueError("Terminate currently only accepts scalar characteristic impedances.")

    if isinstance(solver, AbstractScatteringTerminator):
        
        n_into = model_into.nports
        if model_from.nports != 2 * n_into:
            raise ValueError(
                f"Scattering termination requires a 2N-port terminating into an N-port. "
                f"Got {model_from.nports} and {n_into}."
            )
            
        s_from = model_from.s(frequency, z0=z0)
        s_into = model_into.s(frequency, z0=z0)
        
        # Broadcast scalar z0 to local port shapes
        z0_from = jnp.broadcast_to(jnp.asarray(z0), (model_from.nports,))
        z0_into = jnp.broadcast_to(jnp.asarray(z0), (n_into,))
        
        # Vectorize across the Frequency axis (axis 0 for S-matrices, None for static z0 arrays)
        vmapped_solver = jax.vmap(solver.run, in_axes=(0, None, 0, None))
        solution = vmapped_solver(s_from, z0_from, s_into, z0_into)
        
        return SimulateResult(solution=solution, z0=z0)
        
    elif isinstance(solver, AbstractTransferTerminator):
        
        if model_from.nports != 2 or model_into.nports != 1:
            raise ValueError("Transfer termination currently only supports terminating a 2-port into a 1-port.")
            
        a_from = model_from.a(frequency)
        s_into = model_into.s(frequency, z0=z0)
        
        z0_into = jnp.broadcast_to(jnp.asarray(z0), (1,))
        
        vmapped_solver = jax.vmap(solver.run, in_axes=(0, 0, None))
        solution = vmapped_solver(a_from, s_into, z0_into)
        
        return SimulateResult(solution=solution, z0=z0)
        
    else:
        raise TypeError(f"Unrecognized terminator type: {type(solver)}")