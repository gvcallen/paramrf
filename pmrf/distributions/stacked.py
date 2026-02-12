import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints

class StackedDistribution(dist.Distribution):
    """
    A meta-distribution that combines N independent, potentially heterogeneous 
    scalar distributions into a single vectorized distribution of length N.
    """
    # We use real_vector as a catch-all support, since the underlying 
    # distributions might have mixed supports (e.g., positive vs. real).
    support = constraints.real_vector

    def __init__(self, dists: list[dist.Distribution], validate_args=None):
        self.dists = dists
        
        # Batch shape is empty () because this represents one event.
        # Event shape is (N,) representing the vectorized parameter.
        batch_shape = ()
        event_shape = (len(dists),)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        # 1. Split the JAX PRNG key into N keys, one for each distribution
        keys = jax.random.split(key, len(self.dists))
        
        # 2. Sample each distribution independently
        samples = [d.sample(k, sample_shape) for k, d in zip(keys, self.dists)]
            
        # 3. Stack along the last axis to form the event_shape (N,)
        return jnp.stack(samples, axis=-1)

    def log_prob(self, value):
        # value has shape: broadcast_shape + (N,)
        log_probs = []
        for i, d in enumerate(self.dists):
            # Extract the i-th component's values
            v_i = value[..., i]
            log_probs.append(d.log_prob(v_i))
            
        # Because they are independent, the joint log probability is the sum 
        # of the individual log probabilities: log(P(A) * P(B)) = log(P(A)) + log(P(B))
        return jnp.sum(jnp.stack(log_probs, axis=-1), axis=-1)
        
    @property
    def mean(self):
        return jnp.stack([d.mean for d in self.dists], axis=-1)

    @property
    def variance(self):
        return jnp.stack([d.variance for d in self.dists], axis=-1)
        
    def icdf(self, q):
        q = jnp.asarray(q)
        icdfs = []
        for i, d in enumerate(self.dists):
            # If q was passed as a vector matching the event shape, slice it. 
            # Otherwise, pass the scalar `q` to all sub-distributions.
            q_i = q[..., i] if q.shape and q.shape[-1] == len(self.dists) else q
            icdfs.append(d.icdf(q_i))
        return jnp.stack(icdfs, axis=-1)