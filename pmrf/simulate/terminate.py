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

    Parameters
    ----------
    model_from : Model
        The source Model being terminated.
    model_into : Model
        The load Model to terminate into.
    frequency : Frequency
        The frequency of the results.
    solver : AbstractTerminator
        An instance of a termination algorithm (e.g., LinearFractionalTerminator or MobiusTerminator).
    z0 : ArrayLike, optional
        The characteristic impedance for parameter evaluation, by default 50.0.

    Returns
    -------
    SimulateResult
        A structured result containing the fully terminated network matrix.

    Raises
    ------
    ValueError
        If a non-scalar characteristic impedance (`z0`) is passed, or if the 
        source and load port counts do not satisfy the 2N-to-N requirement.
    TypeError
        If the provided solver does not inherit from `AbstractScatteringTerminator` 
        or `AbstractTransferTerminator`.
    """
    if not jnp.isscalar(z0):
        raise ValueError("Terminate currently only accepts scalar characteristic impedances.")

    n_into = model_into.nports

    if model_from.nports != 2 * n_into:
        raise ValueError(
            f"Termination requires a 2N-port terminating into an N-port. "
            f"Got {model_from.nports} and {n_into}."
        )
        
    if isinstance(solver, AbstractScatteringTerminator):
        s_from = model_from.s(frequency, z0=z0)
        s_into = model_into.s(frequency, z0=z0)
        z0_from = jnp.broadcast_to(jnp.asarray(z0), (model_from.nports,))
        z0_into = jnp.broadcast_to(jnp.asarray(z0), (n_into,))
        
        vmapped_solver = jax.vmap(solver.run, in_axes=(0, None, 0, None))
        solution = vmapped_solver(s_from, z0_from, s_into, z0_into)
        
        return SimulateResult(solution=solution, z0=z0)
        
    elif isinstance(solver, AbstractTransferTerminator):
        a_from = model_from.a(frequency)
        s_into = model_into.s(frequency, z0=z0)
        
        z0_into = jnp.broadcast_to(jnp.asarray(z0), (1,))
        
        vmapped_solver = jax.vmap(solver.run, in_axes=(0, 0, None))
        solution = vmapped_solver(a_from, s_into, z0_into)
        
        return SimulateResult(solution=solution, z0=z0)
    else:
        raise TypeError(f"Unrecognized terminator type: {type(solver)}")