import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
from typing import Sequence

from pmrf.frequency import Frequency
from pmrf.base import AbstractComponent
from pmrf.simulate.base import AbstractCascader, AbstractScatteringCascader, AbstractABCDCascader
from pmrf.simulate.result import SimulateResult

def cascade(
    components: Sequence[AbstractComponent],
    frequency: Frequency,
    solver: AbstractCascader,
    z0: ArrayLike = 50.0,
) -> SimulateResult:
    """
    Cascades a chain of 2N-port components in series.

    Parameters
    ----------
    components : Sequence[AbstractComponent]
        An ordered sequence of components to cascade (left to right).
    frequency : Frequency
        The frequency sweep over which to characterize the network.
    solver : AbstractCascader
        An instance of a cascader algorithm (e.g., Redheffer or TransferCascader).
    z0 : ArrayLike, optional
        The characteristic impedance for parameter evaluation, by default 50.0.

    Returns
    -------
    SimulateResult
        A structured result containing the final cascaded matrices.

    Raises
    ------
    ValueError
        If an empty sequence of components is provided, or if a non-scalar 
        characteristic impedance (`z0`) is passed to a scattering cascader.
    TypeError
        If the provided solver does not inherit from `AbstractScatteringCascader` 
        or `AbstractTransferCascader`.
    """
    if not components:
        raise ValueError("Cannot cascade an empty list of components.")

    if isinstance(solver, AbstractScatteringCascader):
        
        # Directly stack the S-matrices from the component list
        # Transpose to (F, N_component, m, m)
        batched_S = jnp.stack([m.s(frequency, z0=z0) for m in components]).transpose(1, 0, 2, 3)
        
        n_components = len(components)
        m_ports = components[0].nports
        batched_z0 = jnp.broadcast_to(jnp.asarray(z0), (n_components, m_ports))
        
        if not jnp.isscalar(z0):
            raise ValueError("Cascade currently only accepts scalar characteristic impedances")
            
        vmapped_solver = jax.vmap(solver.run, in_axes=(0, None))
        solution = vmapped_solver(batched_S, batched_z0)
        
        return SimulateResult(solution=solution, z0=z0)
        
    elif isinstance(solver, AbstractABCDCascader):
        
        # Directly stack the A-matrices from the component list
        batched_A = jnp.stack([m.a(frequency) for m in components]).transpose(1, 0, 2, 3)
        
        vmapped_solver = jax.vmap(solver.run, in_axes=(0,))
        solution = vmapped_solver(batched_A)
        
        return SimulateResult(solution=solution, z0=z0)
        
    else:
        raise TypeError(f"Unrecognized cascader type: {type(solver)}")