from numpyro.distributions import (
    Distribution,
    Uniform as UniformDistribution,
    LogUniform as LogUniformDistribution,
    Normal as NormalDistribution,
    MultivariateNormal as MultivariateNormalDistribution,
    LogNormal as LogNormalDistribution,
)

from pmrf.distributions.joint import JointDistribution
from pmrf.distributions.stacked import StackedDistribution