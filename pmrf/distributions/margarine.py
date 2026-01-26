from typing import BinaryIO
import os
import jax
import jax.numpy as jnp
import numpy as np
import tempfile

from pmrf.distributions.trainable import TrainableDistribution
from pmrf.distributions.serializable import SerializableDistribution

os.environ["TF_USE_LEGACY_KERAS"] = "True"

class MargarineMAFDistribution(TrainableDistribution, SerializableDistribution):
    """
    Adapter for MAF models from the 'margarine' library.
    Only works for margarine version < 2
    """
    def __init__(self, maf, validate_args=None):
        from margarine.maf import MAF
        self.maf: MAF = maf
        event_shape = (len(self.maf.theta_min),)
        super().__init__(batch_shape=(), event_shape=event_shape, validate_args=validate_args)
        
    def save(self, target: str | BinaryIO):
        if isinstance(target, str):
            return self.maf.save(target)
        return self.write(target)
    
    @classmethod
    def load(cls, source: str | BinaryIO) -> 'MargarineMAFDistribution':
        from margarine.maf import MAF
        if isinstance(source, str):
            return MargarineMAFDistribution(MAF.load(source))
        return cls.read(source)
    
    @classmethod
    def from_samples(cls, samples: jnp.ndarray, construct_kwargs: dict | None = None, **kwargs):
        from margarine.maf import MAF
        construct_kwargs = construct_kwargs or {}
        maf = MAF(samples, **construct_kwargs)
        kwargs.setdefault('epochs', 20000)
        maf.train(**kwargs)
        return MargarineMAFDistribution(maf)
    
    @classmethod
    def from_weighted_samples(cls, samples: jnp.ndarray, weights: jnp.ndarray, construct_kwargs: dict | None = None, **kwargs):
        from margarine.maf import MAF
        construct_kwargs = construct_kwargs or {}
        maf = MAF(samples, weights=weights, **construct_kwargs)
        kwargs.setdefault('epochs', 20000)
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


class MargarineDistribution(TrainableDistribution, SerializableDistribution):
    """
    Adapter for arbitrary models from the 'margarine' library.
    
    Currently only support's RealNVP.
    Only for margarine version >= 2.
    """
    def __init__(self, estimator, validate_args=None):
        from margarine.estimators.realnvp import RealNVP
        self.estimator: RealNVP = estimator
        event_shape = (len(self.estimator.theta),)
        super().__init__(batch_shape=(), event_shape=event_shape, validate_args=validate_args)
        
    def save(self, target: str | BinaryIO):
        if isinstance(target, str):
            return self.estimator.save(target)
        return self.write(target)
    
    @classmethod
    def load(cls, source: str | BinaryIO) -> 'MargarineDistribution':
        from margarine.estimators.realnvp import RealNVP
        if isinstance(source, str):
            return RealNVP(RealNVP.load(source))
        return cls.read(source)
    
    @classmethod
    def from_samples(cls, samples: jnp.ndarray, key=None, init_kwargs: dict | None = None, **train_kwargs):
        from margarine.estimators.realnvp import RealNVP
        if key is None:
            raise Exception('Need key to train Margarine RealNVP')
        
        init_kwargs = init_kwargs or {}
        
        if 'in_size' in init_kwargs and init_kwargs['in_size'] != len(samples):
            raise Exception('In size must be equal to number of sample parameters')
        init_kwargs['in_size'] = len(samples)
        
        estimator = RealNVP(key, samples, **init_kwargs)
        estimator.train(**train_kwargs)
        return MargarineDistribution(estimator)
    
    @classmethod
    def from_weighted_samples(cls, samples: jnp.ndarray, weights: jnp.ndarray, key=None, init_kwargs: dict | None = None, **train_kwargs):
        from margarine.estimators.realnvp import RealNVP
        if key is None:
            raise Exception('Need key to train Margarine RealNVP')        
        
        init_kwargs = init_kwargs or {}
        
        if 'in_size' in init_kwargs and init_kwargs['in_size'] != len(samples):
            raise Exception('In size must be equal to number of sample parameters')
        init_kwargs['in_size'] = len(samples)        
        
        estimator = RealNVP(samples, weights=weights, **init_kwargs)
        estimator.train(key, **train_kwargs)
        return MargarineDistribution(estimator)

    def sample(self, key, sample_shape):
        n_samples = int(jnp.prod(jnp.array(sample_shape))) or 1
        samples = self.estimator.sample(key, num_samples=n_samples)
        samples = jnp.array(samples, dtype=jnp.float32)
        return samples.reshape(sample_shape + self.batch_shape + self.event_shape)

    def log_prob(self, value):
        value = jnp.atleast_2d(value)
        return self.estimator.log_prob(value).astype(np.float32)

    def icdf(self, u):
        u = jnp.atleast_2d(u)
        return self.estimator(u)