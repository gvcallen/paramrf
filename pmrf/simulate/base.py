"""pmrf/simulate/base.py"""

import numpy as np
from abc import abstractmethod
import equinox as eqx
import jax.numpy as jnp
from typing import Any


class NodalRepresentation(eqx.Module):
    num_nodes: int = eqx.field(static=True)
    r_idx: np.ndarray
    c_idx: np.ndarray
    ext_idx: np.ndarray
    int_idx: np.ndarray

class PortRepresentation(eqx.Module):
    num_ports: int = eqx.field(static=True)
    ext_idx: np.ndarray = eqx.field(static=True)
    int_idx: np.ndarray = eqx.field(static=True)
    
    # 1D array of length `num_ports`.
    # Value is the integer ID of the Net/Node that port connects to.
    port_to_net_map: np.ndarray = eqx.field(static=True)

class ScatteringResult(eqx.Module):
    s: jnp.ndarray
    z0: jnp.ndarray
    
    success: bool = True
    metrics: Any = None

class AdmittanceResult(eqx.Module):
    y: jnp.ndarray
    
    success: bool = True
    metrics: Any = None


class AbstractAdmittanceReducer(eqx.Module):
    @abstractmethod
    def run(
        self, 
        y_matrices: jnp.ndarray,
        topology: NodalRepresentation, 
    ) -> AdmittanceResult:
        raise NotImplementedError
    
class AbstractScatteringReducer(eqx.Module):
    #: The required layout of the incoming S-matrices, either 'block_diagonal' or 'stacked'
    s_layout: eqx.AbstractClassVar[str]
    
    @abstractmethod
    def run(
        self, 
        s_matrices: jnp.ndarray,
        port_z0: jnp.ndarray,
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        raise NotImplementedError


AbstractReducer = AbstractAdmittanceReducer | AbstractScatteringReducer