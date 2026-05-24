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
    ext_idx: np.ndarray
    int_idx: np.ndarray
    
    # 1D array of length `num_ports`.
    # Value is the integer ID of the Net/Node that port connects to.
    port_to_net_map: np.ndarray

class ScatteringResult(eqx.Module):
    s: jnp.ndarray
    z0: jnp.ndarray
    
    success: bool = True
    metrics: Any = None

class AdmittanceResult(eqx.Module):
    y: jnp.ndarray
    
    success: bool = True
    metrics: Any = None

class TransferResult(eqx.Module):
    a: jnp.ndarray
    
    success: bool = True
    metrics: Any = None

class AbstractAdmittanceReducer(eqx.Module):
    @abstractmethod
    def run(
        self, 
        y_flattened: jnp.ndarray,
        topology: NodalRepresentation, 
    ) -> AdmittanceResult:
        raise NotImplementedError
    
class AbstractScatteringReducer(eqx.Module):
    @abstractmethod
    def run(
        self, 
        s_block_diagonal: jnp.ndarray,
        port_z0: jnp.ndarray,
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        raise NotImplementedError
    
class AbstractScatteringCascader(eqx.Module):
    @abstractmethod
    def run(
        self, 
        s_stacked: jnp.ndarray,
        port_z0: jnp.ndarray,
    ) -> ScatteringResult:
        raise NotImplementedError
    
class AbstractTransferCascader(eqx.Module):
    @abstractmethod
    def run(
        self, 
        a_stacked: jnp.ndarray,
    ) -> TransferResult:
        raise NotImplementedError
    
class AbstractScatteringTerminator(eqx.Module):
    @abstractmethod
    def run(
        self, 
        s_from: jnp.ndarray, 
        z0_from: jnp.ndarray, 
        s_into: jnp.ndarray, 
        z0_into: jnp.ndarray
    ) -> ScatteringResult:
        raise NotImplementedError

class AbstractTransferTerminator(eqx.Module):
    @abstractmethod
    def run(
        self, 
        a_from: jnp.ndarray, 
        s_into: jnp.ndarray, 
        z0_into: jnp.ndarray
    ) -> ScatteringResult:
        raise NotImplementedError

AbstractReducer = AbstractAdmittanceReducer | AbstractScatteringReducer
AbstractCascader = AbstractScatteringCascader | AbstractTransferCascader
AbstractTerminator = AbstractScatteringTerminator | AbstractTransferTerminator