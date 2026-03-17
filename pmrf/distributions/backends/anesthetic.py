import io

import jax
import jax.numpy as jnp

from numpyro.distributions import Distribution

class AnestheticDistribution(Distribution):
    """
    Adapter for distributions represented by samples from the Anesthetic library.
    
    Although you cannot sample from the nested samples as if they were an actual distribution,
    this class is still useful in storing sampling results directly within a model.
    """
    def __init__(self, nested_samples, param_names: list[str] | None = None, validate_args=None):
        from anesthetic import NestedSamples
        
        if param_names:
            nested_samples = nested_samples.loc[:, param_names + ['logL', 'logL_birth', 'nlive']]
        else:
            columns = nested_samples.columns
            param_names = [columns[i][0] for i in range(len(columns))]
            param_names = [name for name in param_names if name not in {'logL', 'logL_birth', 'nlive'}]
        
        self.nested_samples: NestedSamples = nested_samples
        
        event_shape = (len(param_names),)
        super().__init__(batch_shape=(), event_shape=event_shape, validate_args=validate_args)
        
    def __getstate__(self):
        """Package the state for pickling."""
        state = self.__dict__.copy()
        state['_csv_payload'] = self.nested_samples.to_csv()
        del state['nested_samples']
        return state

    def __setstate__(self, state):
        """Unpack the state from pickling."""
        from anesthetic import NestedSamples, read_csv
        csv_payload = state.pop('_csv_payload')
        self.__dict__.update(state)
        self.nested_samples = NestedSamples(read_csv(io.StringIO(csv_payload)))

    def sample(self, key, sample_shape=()):
        # Ensure sample_shape is a tuple
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
            
        available_samples = self.samples()
        num_available = available_samples.shape[0]
        
        # Strategy: Sample uniformly with replacement using random integers.
        # This allows us to draw any number of samples (larger or smaller than num_available)
        # from the fixed underlying empirical distribution without running out of samples.
        indices = jax.random.randint(
            key, 
            shape=sample_shape, 
            minval=0, 
            maxval=num_available
        )
        
        return available_samples[indices]

    
    def param_names(self) -> list[str]:
        columns = self.nested_samples.columns
        param_names = [columns[i][0] for i in range(len(columns))]
        param_names = [name for name in param_names if name not in {'logL', 'logL_birth', 'nlive'}]        
        return param_names
    
    def samples(self, prior=False, weighted=False) -> jnp.ndarray:
        if weighted:
            if prior:
                nested_samples = self.nested_samples.prior()
            else:
                nested_samples = self.nested_samples
        else:
            if prior:
                nested_samples = self.nested_samples.prior_points()
            else:
                nested_samples = self.nested_samples.posterior_points()
        
        prior_samples = nested_samples.loc[:, self.param_names()].to_numpy()
        return jnp.array(prior_samples)
    
    def weights(self, prior=False) -> jnp.ndarray:
        if prior:
            return jnp.array(self.nested_samples.prior().get_weights())
        else:
            return jnp.array(self.nested_samples.get_weights())
        
    def bounds(self, prior=False) -> tuple[jnp.ndarray, jnp.ndarray]:
        samples = self.samples(prior=prior)
        
        return (jnp.min(samples, axis=0), jnp.max(samples, axis=0))
    
    def marginalize(self, keep_params: list[str]) -> "AnestheticDistribution":
        """
        Marginalizes out all parameters except those specified in keep_params.
        
        Parameters
        ----------
        keep_params : list[str]
            The list of parameter names to retain. All other parameters 
            will be marginalized out.
            
        Returns
        -------
        AnestheticDistribution
            A new distribution object representing the marginalized distribution.
        """
        # Verify the requested parameters actually exist in the current distribution
        current_params = self.param_names()
        missing = [p for p in keep_params if p not in current_params]
        if missing:
            raise ValueError(f"Cannot keep parameters that don't exist: {missing}")
            
        # Return a new instance using the exact same underlying nested_samples DataFrame,
        # but restricted to the new, smaller subset of param_names.
        return AnestheticDistribution(
            nested_samples=self.nested_samples,
            param_names=keep_params,
            validate_args=self._validate_args
        )    