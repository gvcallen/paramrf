import os
import jax
import jax.numpy as jnp
import numpy as np

from numpyro.distributions import Distribution

os.environ["TF_USE_LEGACY_KERAS"] = "True"

from margarine.maf import MAF

class MargarineMAFDistribution(Distribution):
    """
    Adapter for MAF models from the 'margarine' library.
    """
    def __init__(self, maf, validate_args=None):
        self.maf: MAF = maf
        event_shape = (len(self.maf.theta_min),)
        super().__init__(batch_shape=(), event_shape=event_shape, validate_args=validate_args)

    def save(self, path: str):
        return self.maf.save(path)
    
    @staticmethod
    def load(path: str):
        return MAF.load(path)
    
    @staticmethod
    def generate(data, weights=None, construct_kwargs: dict = {}, **kwargs):
        from margarine.maf import MAF
        if weights is not None:
            maf = MAF(data, weights=weights, **construct_kwargs)
        else:
            maf = MAF(data, **construct_kwargs)
        maf.train(**kwargs)

        return MargarineMAFDistribution(maf)

    def sample(self, key, sample_shape):
        n_samples = int(jnp.prod(jnp.array(sample_shape))) or 1
        samples = self.maf.sample(length=n_samples)
        samples = jnp.array(samples, dtype=jnp.float32)
        return samples.reshape(sample_shape + self.batch_shape + self.event_shape)

    def log_prob(self, value):
        value = jnp.atleast_2d(value)
        def compute_logp(x_np):
            return self.maf.log_prob(x_np).astype(np.float32)

        return jax.pure_callback(
            compute_logp,
            jax.ShapeDtypeStruct((value.shape[0],), jnp.float32),
            value
        )

    def icdf(self, u):
        import tensorflow as tf
        u = jnp.atleast_2d(u)
        def tf_wrapper(u_np):
            x_tf = tf.convert_to_tensor(u_np, dtype=tf.float32)
            result = self.maf(x_tf)  # Assumes maf(u) maps uniform→target
            return np.array(result, dtype=np.float32)

        return jax.pure_callback(
            tf_wrapper,
            jax.ShapeDtypeStruct(u.shape, jnp.float32),
            u
        )

    @property
    def min(self):
        return jnp.array(self.maf.theta_min, dtype=jnp.float32)

    @property
    def max(self):
        return jnp.array(self.maf.theta_max, dtype=jnp.float32)
