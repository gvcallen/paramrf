import io

import jax.numpy as jnp

from pmrf.distributions.sampled import SampledDistribution

class AnestheticDistribution(SampledDistribution):
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
    
    def param_names(self) -> list[str]:
        columns = self.nested_samples.columns
        param_names = [columns[i][0] for i in range(len(columns))]
        param_names = [name for name in param_names if name not in {'logL', 'logL_birth', 'nlive'}]        
        return param_names
    
    def sample(self, key, sample_shape):
        raise NotImplementedError    

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