import dataclasses
from dataclasses import dataclass

import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions.distribution import Distribution

from pmrf.core.field import field
from pmrf.core.parameter import Parameter, MIN_PERCENTILE, MAX_PERCENTILE

@dataclass
class ParameterGroup:
    r"""
    A metadata class that groups a set of named flat parameters and defines any relationships between them.

    Attributes
    ----------
    parameter_names : list[str]
        The names of the parameters included in this group.
    distribution : dist.Distribution or None
        An optional joint distribution over the flattened parameters.
    """
    param_names: list[str]
    distribution: dist.Distribution | None = field(default=None)
    
    def __init__(self, param_names: list[str] | dict[str, Parameter], distribution: dist.Distribution | None = None):
        r"""
        Construct a :class:`ParameterGroup`.

        Parameters
        ----------
        param_names : list[str] | dict[str, Parameter]
            The names of the flattened parameters (or a mapping to parameters).
        dist : numpyro.distributions.Distribution, optional
            An optional joint distribution over the flattened parameters.
        """
        self.param_names = param_names
        self.distribution = distribution
        
    @property
    def num_params(self):
        r"""
        Number of flattened parameters in the group.

        Returns
        -------
        int
            The count of names in ``parameter_names``.
        """
        return len(self.param_names)
            
    @property
    def min(self) -> jnp.array:
        r"""
        The unscaled minimum value of the parameter group's distribution.
        
        Determined by the `MIN_PERCENTILE` quantile.

        Returns
        -------
        jnp.array
            The minimum value, or -inf if no distribution is set.
        """
        if self.distribution is not None:
            if hasattr(self.distribution, 'min'):
                return self.distribution.min.reshape((self.num_params))
            elif hasattr(self.distribution, 'low'):
                return self.distribution.low.reshape((self.num_params))
            else:
                return self.distribution.icdf(jnp.array([MIN_PERCENTILE] * self.num_params))
            
        return jnp.array([-jnp.inf] * self.num_params)
    
    @property
    def max(self) -> jnp.array:
        r"""
        The unscaled maximum value of the parameter group's distribution.
        
        Determined by the `MAX_PERCENTILE` quantile.

        Returns
        -------
        jnp.array
            The maximum value, or inf if no distribution is set.
        """
        if self.distribution is not None:
            if hasattr(self.distribution, 'max'):
                return self.distribution.max.reshape((self.num_params))
            elif hasattr(self.distribution, 'high'):
                return self.distribution.high.reshape((self.num_params))
            else:
                # TODO implement optimization to determine maximum
                return self.distribution.icdf(jnp.array([MAX_PERCENTILE] * self.num_params))
            
        return jnp.array([jnp.inf] * self.num_params)
    
    def with_distribution(self, distribution: Distribution) -> 'Parameter':
        r"""
        Return a copy of the parameter group with a new distribution.

        Parameters
        ----------
        distribution : numpyro.distributions.Distribution
            The distribution to associate with this parameter.

        Returns
        -------
        Parameter
            A copy of this object with ``distribution`` replaced.

        Raises
        ------
        Exception
            If ``dist`` is not a numpyro Distribution.
        """
        if not isinstance(distribution, Distribution):
            raise Exception('Only numpyro distributions are supported as parameter distributions')
        
        return dataclasses.replace(self, distribution=distribution)