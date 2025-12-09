from numpyro.distributions import (
    Distribution,
    Uniform as UniformDistribution,
    LogUniform as LogUniformDistribution,
    Normal as NormalDistribution,
    MultivariateNormal as MultivariateNormalDistribution,
    LogNormal as LogNormalDistribution,
)

from pmrf.distributions.trainable import TrainableDistribution, TrainableDistributionT

try:
    from pmrf.distributions.margarine import MargarineMAFDistribution
except ImportError:
    pass

try:
    from pmrf.distributions.flowjax import FlowJAXDistribution
except ImportError:
    pass

from pmrf.distributions.parameter import JointParameterDistribution