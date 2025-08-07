import os
import jax
import jax.numpy as jnp
import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "True"

from margarine.maf import MAF

class MargarineMAFAdapter:
    """
    Adapter for MAF models from the 'margarine' library.
    """
    def __init__(self, maf):
        self.maf: MAF = maf
        self.event_dim = len(self.maf.theta_min)

    def save(self, path: str):
        return self.maf.save(path)
    
    @staticmethod
    def load(path: str):
        return MAF.load(path)
    
    @staticmethod
    def generate(data, weights=None, construct_kwargs: dict = {}, **kwargs):
        from margarine.maf import MAF
        maf = MAF(data, weights=weights, **construct_kwargs)
        maf.train(**kwargs)
        return maf

    def sample(self, key, sample_shape):
        n_samples = int(jnp.prod(jnp.array(sample_shape))) or 1
        samples = self.maf.sample(length=n_samples)
        samples = jnp.array(samples, dtype=jnp.float32)
        return samples.reshape(sample_shape + (self.event_dim,))

    def log_prob(self, value):
        value = jnp.atleast_2d(value)
        if value.shape[-1] != self.event_dim:
            raise ValueError(f"log_prob expected last dimension {self.event_dim}, got {value.shape[-1]}")

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
        if u.shape[-1] != self.event_dim:
            raise ValueError(f"icdf expected last dimension {self.event_dim}, got {u.shape[-1]}")

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
