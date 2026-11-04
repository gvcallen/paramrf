"""
Probability distribution classes and factories.

Can be used for random parameters in :mod:`pmrf.parameters`.

These are (mostly) re-exports from `parax <https://gvcallen.github.io/parax>`_, which in turn
takes them from the `distreqx <https://lockwo.github.io/distreqx>`_ library, filling in any that
the installed version of `distreqx` does not provide. The goal is to cover the most common
applications; for more advanced use-cases, simply use `distreqx` directly instead.
"""
import jax.numpy as jnp
import jax
import equinox as eqx
from jaxtyping import ArrayLike

from parax.distributions import (
    AbstractDistribution as AbstractDistribution,
    AbstractSTDDistribution as AbstractSTDDistribution,
    AbstractSurvivalDistribution as AbstractSurvivalDistribution,
    Gamma as Gamma,
    Joint as Joint,
    LogNormal as LogNormal,
    Normal as Normal,
    Transformed as Transformed,
    TruncatedNormal as TruncatedNormal,
    Uniform as Uniform,
)

from parax.probability import truncate_distribution as truncate


class LogUniform(AbstractSTDDistribution, AbstractSurvivalDistribution):
    r"""Log-uniform distribution on the positive interval $[low, high]$.

    **Mathematical Formulation**

    $$p(x) = 1 / (x \ln(high / low))$$ for $low \le x \le high$.
    Equal intervals in log space have equal probability.

    References
    ----------
    Jeffreys, H. (1946). An invariant form for the prior probability in
    estimation problems. Proceedings of the Royal Society A, 186, 453–461.
    """

    #: Lower support bound.
    low: jax.Array = eqx.field(converter=jnp.asarray)
    #: Upper support bound.
    high: jax.Array = eqx.field(converter=jnp.asarray)

    def __init__(self, low: ArrayLike, high: ArrayLike):
        dtype = jnp.result_type(low, high, float)
        low, high = jnp.asarray(low, dtype=dtype), jnp.asarray(high, dtype=dtype)
        if not bool(jnp.all((low > 0) & (low < high) & jnp.isfinite(high))):
            raise ValueError("LogUniform requires 0 < low < high with finite bounds")
        self.low = low
        self.high = high

    @property
    def event_shape(self):
        return self.low.shape

    @property
    def support(self):
        return self.low, self.high

    @property
    def log_range(self):
        return jnp.log(self.high) - jnp.log(self.low)

    def log_prob(self, value):
        value = jnp.asarray(value)
        return jnp.where(
            (value >= self.low) & (value <= self.high),
            -jnp.log(value) - jnp.log(self.log_range),
            -jnp.inf,
        )

    def prob(self, value):
        return jnp.exp(self.log_prob(value))

    def sample(self, key):
        u = jax.random.uniform(key, shape=self.low.shape, dtype=self.low.dtype)
        return self.icdf(u)

    def sample_and_log_prob(self, key):
        sample = self.sample(key)
        return sample, self.log_prob(sample)

    def icdf(self, value):
        result = jnp.exp(jnp.log(self.low) + value * self.log_range)
        return jnp.where(value == 0, self.low,
                         jnp.where(value == 1, self.high, result))

    def cdf(self, value):
        value = jnp.asarray(value)
        return jnp.clip((jnp.log(jnp.maximum(value, self.low)) - jnp.log(self.low))
                        / self.log_range, 0.0, 1.0)

    def log_cdf(self, value):
        return jnp.log(self.cdf(value))

    def mean(self):
        return (self.high - self.low) / self.log_range

    def median(self):
        return jnp.exp((jnp.log(self.low) + jnp.log(self.high)) / 2)

    def variance(self):
        second_moment = (self.high ** 2 - self.low ** 2) / (2 * self.log_range)
        return second_moment - self.mean() ** 2

    def entropy(self):
        return (jnp.log(self.low) + jnp.log(self.high)) / 2 + jnp.log(self.log_range)

    def mode(self):
        return self.low

    def kl_divergence(self, other_dist, **kwargs):
        if not isinstance(other_dist, LogUniform):
            raise NotImplementedError("KL divergence is only implemented between LogUniform distributions")
        return jnp.where(
            (other_dist.low <= self.low) & (self.high <= other_dist.high),
            jnp.log(other_dist.log_range / self.log_range),
            jnp.inf,
        )

def CenteredUniform(
    center: ArrayLike, 
    half_width: ArrayLike, 
) -> Uniform:
    """
    Create a Uniform distribution defined by center and width.
    """
    center, half_width = jnp.asarray(center), jnp.asarray(half_width)

    lower = center - half_width
    upper = center + half_width
    return Uniform(lower, upper)


def RelativeNormal(
    mean: ArrayLike, 
    pct_std: ArrayLike, 
) -> AbstractDistribution:
    """
    Create a Normal distribution defined by relative standard deviation.
    """
    mean, pct_std = jnp.asarray(mean), jnp.asarray(pct_std)

    std = mean * pct_std
    return Normal(mean, std)


def RelativeTruncatedNormal(
    mean: ArrayLike,
    pct_std: ArrayLike,
    trunc_std: ArrayLike = 3.0,
) -> AbstractDistribution:
    """
    Create a symmetric Truncated Normal distribution defined by relative standard deviation
    and truncated at a specified number of standard deviations from the mean.
    """
    mean, pct_std, trunc_std = (
        jnp.asarray(mean), 
        jnp.asarray(pct_std), 
        jnp.asarray(trunc_std)
    )

    std = mean * pct_std
    low = mean - (trunc_std * std)
    high = mean + (trunc_std * std)
    
    return TruncatedNormal(loc=mean, scale=std, low=low, high=high)


__all__ = [
    "AbstractDistribution",
    "CenteredUniform",
    "Gamma",
    "Joint",
    "LogUniform",
    "LogNormal",
    "Normal",
    "RelativeNormal",
    "RelativeTruncatedNormal",
    "Transformed",
    "TruncatedNormal",
    "Uniform",
    "truncate",
]
