"""
A distribution composed of multiple independent distributions.
"""

from numpyro.distributions import Distribution, constraints
import numpyro.distributions as dist
import jax.numpy as jnp
import jax

from pmrf.constants import MIN_PERCENTILE, MAX_PERCENTILE

class JointDistribution(Distribution):
    support = constraints.real_vector
    reparametrized_params = []
    has_rsample = True

    def __init__(
        self, 
        distributions: list[Distribution], 
        distribution_names: list[list[str]], 
        param_names: list[str]
    ):
        """
        A joint distribution composed of multiple independent component distributions.

        Parameters
        ----------
        distributions : list[Distribution]
            List of numpyro distributions.
        distribution_names : list[list[str]]
            List of lists, where each sub-list contains the names of the parameters 
            associated with the corresponding distribution.
        param_names : list[str]
            The global ordered list of all parameter names.
        """
        self.distributions = distributions
        self.distribution_names = distribution_names
        self.param_names = param_names
        self.name_to_index = {name: i for i, name in enumerate(param_names)}
        
        self._validate_inputs()

        event_shape = (len(param_names),)
        batch_shape = ()
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def _validate_inputs(self):
        if len(self.distributions) != len(self.distribution_names):
            raise ValueError("The number of distributions must match the number of name lists in distribution_names.")
            
        all_provided_names = [name for names in self.distribution_names for name in names]
        if set(all_provided_names) != set(self.param_names):
            raise ValueError("Mismatch between distribution_names and the global param_names.")

        for dist in self.distributions:
            if not hasattr(dist, "sample") or not hasattr(dist, "log_prob"):
                raise ValueError("All distributions must support 'sample' and 'log_prob'.")

    @property
    def num_params(self) -> int:
        return len(self.param_names)

    def sample(self, key, sample_shape=()):
        values = [None] * len(self.param_names)
        for dist, group_names in zip(self.distributions, self.distribution_names):
            subkey, key = jax.random.split(key)
            samples = dist.sample(subkey, sample_shape)
            
            # Expand dimension if the distribution is univariate to allow [..., i] indexing
            if not dist.event_shape:
                samples = samples[..., None]
                
            for i, name in enumerate(group_names):
                idx = self.name_to_index[name]
                values[idx] = samples[..., i]
        return jnp.stack(values, axis=-1)

    def log_prob(self, value):
        total_logp = 0.0
        for dist, group_names in zip(self.distributions, self.distribution_names):
            idxs = [self.name_to_index[name] for name in group_names]
            group_vals = value[..., idxs]
            
            # Squeeze dimension if the distribution expects scalars
            if not dist.event_shape and len(group_names) == 1:
                group_vals = group_vals[..., 0]
                
            logp = dist.log_prob(group_vals)
            total_logp += logp
        return total_logp

    def icdf(self, u):
        values = [None] * len(self.param_names)
        for dist, group_names in zip(self.distributions, self.distribution_names):
            idxs = [self.name_to_index[name] for name in group_names]
            group_u = u[..., idxs]
            
            if not dist.event_shape and len(group_names) == 1:
                group_u = group_u[..., 0]
                
            group_x = dist.icdf(group_u)
            
            if not dist.event_shape:
                group_x = group_x[..., None]
                
            for i, name in enumerate(group_names):
                values[self.name_to_index[name]] = group_x[..., i]
        return jnp.stack(values, axis=-1)

    def cdf(self, x):
        values = [None] * len(self.param_names)
        for dist, group_names in zip(self.distributions, self.distribution_names):
            idxs = [self.name_to_index[name] for name in group_names]
            group_x = x[..., idxs]
            
            if not dist.event_shape and len(group_names) == 1:
                group_x = group_x[..., 0]
                
            group_u = dist.cdf(group_x)
            
            if not dist.event_shape:
                group_u = group_u[..., None]
                
            for i, name in enumerate(group_names):
                values[self.name_to_index[name]] = group_u[..., i]
        return jnp.stack(values, axis=-1)
    
    @property
    def min(self) -> jnp.ndarray:
        """
        Dynamically infer lower bounds based on the numpyro constraints of each distribution.
        
        MIN_PERCENTILE is used if the distribution is not strictly bounded.
        """
        lower_bounds = [None] * len(self.param_names)
        
        for d, group_names in zip(self.distributions, self.distribution_names):
            # 1. Safely extract the lower bound for this distribution group
            if isinstance(d, dist.ImproperUniform):
                lb_val = -jnp.inf
            elif isinstance(d, dist.Delta):
                lb_val = d.v
            elif isinstance(d, dist.Uniform):
                lb_val = d.low
            else:
                # Best-effort extraction from Numpyro constraints
                lb_val = getattr(d.support, "lower_bound", d.icdf(MIN_PERCENTILE))
            
            # 2. Map the extracted bounds back to the global parameter order
            for i, name in enumerate(group_names):
                idx = self.name_to_index[name]
                
                lb = lb_val
                # Handle cases where the bound is a vector (e.g., multivariate constraints)
                if isinstance(lb, (jnp.ndarray, float, int)) and jnp.ndim(lb) > 0:
                    lb = lb[i]

                lower_bounds[idx] = lb

        return jnp.array(lower_bounds)

    @property
    def max(self) -> jnp.ndarray:
        """
        Dynamically infer upper bounds based on the numpyro constraints of each distribution.
        
        MAX_PERCENTILE is used if the distribution is not strictly bounded.
        """
        upper_bounds = [None] * len(self.param_names)
        
        for d, group_names in zip(self.distributions, self.distribution_names):
            # 1. Safely extract the upper bound for this distribution group
            if isinstance(d, dist.ImproperUniform):
                ub_val = jnp.inf
            elif isinstance(d, dist.Delta):
                ub_val = d.v
            elif isinstance(d, dist.Uniform):
                ub_val = d.high
            else:
                # Best-effort extraction from Numpyro constraints
                ub_val = getattr(d.support, "upper_bound", d.icdf(MAX_PERCENTILE))
            
            # 2. Map the extracted bounds back to the global parameter order
            for i, name in enumerate(group_names):
                idx = self.name_to_index[name]
                
                ub = ub_val
                # Handle cases where the bound is a vector (e.g., multivariate constraints)
                if isinstance(ub, (jnp.ndarray, float, int)) and jnp.ndim(ub) > 0:
                    ub = ub[i]

                upper_bounds[idx] = ub

        return jnp.array(upper_bounds)    

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Dynamically infer bounds based on the numpyro constraints of each distribution.
        
        MIN_PERCENTILE and MAX_PERCENTILE are used for "min" and "max" if the distribution
        is not bounded.
        """
        return self.min, self.max