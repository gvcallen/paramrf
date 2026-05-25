"""
Base adapter models used as abstract bases for concrete adapters.
"""

from abc import ABC, abstractmethod

import numpy as np
import jax
import jax.numpy as jnp

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.utils.type import is_overridden
from pmrf.utils import field, freeze
from pmrf.types import ArrayLike
from pmrf.rf import renormalize_s

class AbstractDiscrete(Model, ABC):
    """
    (experimental) A model whose properties are defined on a discrete (tabulated) frequency grid.
    
    To use, set `self.frequency` and override one or more of the `xxx_discrete` methods.
    The base Model conversions (s2a, s2z, etc.) will be applied automatically
    to the interpolated values.

    Parameters
    ----------
    frequency : Frequency
        The constant frequency over which the discrete model is defined.
    """
    #: The constant frequency.
    frequency: Frequency = field(converter=freeze)

    # Tabulated data entry points
    def s_discrete(self, z0: ArrayLike = 50.0) -> jnp.ndarray: raise NotImplementedError
    def a_discrete(self) -> jnp.ndarray: raise NotImplementedError
    def y_discrete(self) -> jnp.ndarray: raise NotImplementedError
    def z_discrete(self) -> jnp.ndarray: raise NotImplementedError

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        if is_overridden(type(self), AbstractDiscrete, 's_discrete'):
            s_intrinsic_old = self.s_discrete(z0=50.0)
            s_intrinsic_new = self._interp(s_intrinsic_old, freq)
            z0_req = jnp.asarray(z0)
            is_matched = jnp.all(z0_req == 50.0)
            
            return jax.lax.cond(
                is_matched,
                lambda: s_intrinsic_new,
                lambda: renormalize_s(s_intrinsic_new, 50.0, z0_req, 'power', 'power')
            )
        return super().s(freq, z0=z0)

    def a(self, freq: Frequency) -> jnp.ndarray:
        if is_overridden(type(self), AbstractDiscrete, 'a_discrete'):
            return self._interp(self.a_discrete(), freq)
        return super().a(freq)

    def y(self, freq: Frequency) -> jnp.ndarray:
        if is_overridden(type(self), AbstractDiscrete, 'y_discrete'):
            return self._interp(self.y_discrete(), freq)
        return super().y(freq)

    def z(self, freq: Frequency) -> jnp.ndarray:
        if is_overridden(type(self), AbstractDiscrete, 'z_discrete'):
            return self._interp(self.z_discrete(), freq)
        return super().z(freq)

    def _interp(self, x: jnp.ndarray, freq: Frequency) -> jnp.ndarray:
        """
        Vectorized interpolation across port matrices without moveaxis.
        Safely handles complex RF matrices.
        """
        f_new = freq.f_scaled
        f_old = self.frequency.f_scaled
        
        def interp_trace(trace):
            # jnp.interp only supports real floats. We must split and recombine.
            trace_real = jnp.interp(f_new, f_old, jnp.real(trace))
            trace_imag = jnp.interp(f_new, f_old, jnp.imag(trace))
            return trace_real + 1j * trace_imag

        # vmap over columns (axis 1)
        vmap_cols = jax.vmap(interp_trace, in_axes=1, out_axes=1)
        # vmap over rows (axis 1 of the remaining array)
        vmap_matrix = jax.vmap(vmap_cols, in_axes=1, out_axes=1)
        
        return vmap_matrix(x)


class AbstractSingleDomain(Model, ABC):
    """
    (experimental) Base model wrapping a single known domain type.
    
    This class handles dynamic domain injection and automatically renormalizes
    S-parameters if the requested characteristic impedance differs from the 
    intrinsic impedance.

    Parameters
    ----------
    domain : str, default='s'
        The domain matrix type (e.g., 's', 'a', 'y', 'z').
    z0 : numpy.ndarray | None, default=50.0
        The characteristic impedance the intrinsic matrix is defined in. 
        Must not be None if `domain` is 's'.
    """
    
    #: The domain matrix type (e.g., 's', 'a', 'y', 'z').
    domain: str = field(default='s', static=True, kw_only=True)
    
    #: Intrinsic characteristic impedance for S-parameters.
    z0: np.ndarray | None = field(
        default=50.0, 
        converter=lambda x: np.asarray(x) if x is not None else None,
        kw_only=True
    )

    @abstractmethod    
    def matrix(self, freq: Frequency) -> jnp.ndarray:
        """
        Compute the intrinsic matrix for the domain.

        Parameters
        ----------
        freq : Frequency
            The frequency grid to evaluate on.

        Returns
        -------
        jax.numpy.ndarray
            The intrinsic domain matrix.
        """
        raise NotImplementedError

    @property
    def primary_domain(self) -> str:
        """
        Get the primary domain string.
        
        Returns
        -------
        str
        """
        return self.domain    
    
    def primary_matrix(self, freq: Frequency, **kwargs) -> jnp.ndarray:
        """
        Retrieve the primary matrix and apply impedance renormalization if necessary.

        Parameters
        ----------
        freq : Frequency
            The frequency grid.
        **kwargs
            Additional arguments, specifically `z0` for S-parameter renormalization.

        Returns
        -------
        jax.numpy.ndarray
            The domain matrix, renormalized to the requested z0 if applicable.
        """
        mat = self.matrix(freq)
        
        if self.domain == 's':
            if self.z0 is None:
                raise ValueError("z0 cannot be None when domain is 's'")
                
            z0_req = jnp.asarray(kwargs.pop('z0', 50.0))
            z0_intrinsic = self.z0
            
            is_matched = jnp.all(z0_req == z0_intrinsic)
            
            return jax.lax.cond(
                is_matched,
                lambda: mat,
                lambda: renormalize_s(mat, z0_intrinsic, z0_req, 'power', 'power')
            )
        return mat


class AbstractSingleDiscreteDomain(AbstractSingleDomain, AbstractDiscrete, ABC):
    """
    (experimental) Base model providing a single dynamic domain from a discrete grid.
    
    To use this class, inherit from it and implement `discrete_matrix` to return 
    the intrinsic tabulated data. Interpolation and impedance renormalization 
    are handled automatically.
    """    
    @abstractmethod
    def discrete_matrix(self) -> jnp.ndarray:
        """
        Return the static discrete matrix.

        Returns
        -------
        jax.numpy.ndarray
            The tabulated intrinsic domain matrix.
        """
        raise NotImplementedError
    
    def matrix(self, freq: Frequency) -> jnp.ndarray:
        """
        Interpolate the discrete matrix onto the requested continuous frequency grid.

        Parameters
        ----------
        freq : Frequency
            The target continuous frequency grid.

        Returns
        -------
        jax.numpy.ndarray
            The interpolated domain matrix.
        """
        return self._interp(self.discrete_matrix(), freq)