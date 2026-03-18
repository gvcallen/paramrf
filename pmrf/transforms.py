import numpyro.distributions.transforms as transforms
from numpyro.distributions import Distribution

class HypercubeTransform(transforms.ParameterFreeTransform):
    domain = transforms.constraints.real_vector
    codomain = transforms.constraints.unit_interval
    dist: Distribution

    def __init__(self, dist: Distribution):
        self.dist = dist
        super().__init__()

    def __call__(self, x):
        return self.dist.cdf(x)

    def _inverse(self, u):
        return self.dist.icdf(u)

def SigmoidHypercubeTransform(dist: Distribution):
    return transforms.ComposeTransform([
        HypercubeTransform(dist),
        transforms.SigmoidTransform().inv
    ])        