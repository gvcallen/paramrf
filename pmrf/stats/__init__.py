"""Public stats modules and their common entry points."""

from importlib import import_module
from typing import TYPE_CHECKING

# Keep static analyzers aware of the public names without loading them at runtime.
if TYPE_CHECKING:
    from .bijectors import (
        AbstractBijector as AbstractBijector,
        Chain as Chain,
        DiagLinear as DiagLinear,
        Exp as Exp,
        Identity as Identity,
        Inverse as Inverse,
        Leafwise as Leafwise,
        Permute as Permute,
        R2ToComplex as R2ToComplex,
        ScalarAffine as ScalarAffine,
        Shift as Shift,
        Sigmoid as Sigmoid,
        Softplus as Softplus,
        Tanh as Tanh,
        Transpose as Transpose,
        TriangularLinear as TriangularLinear,
    )
    from .covariance_kernels import (
        AbstractCovarianceKernel as AbstractCovarianceKernel,
        AutoCrossKernel as AutoCrossKernel,
        ConstantKernel as ConstantKernel,
        CosineKernel as CosineKernel,
        Matern32Kernel as Matern32Kernel,
        Matern52Kernel as Matern52Kernel,
        PeriodicKernel as PeriodicKernel,
        ProductKernel as ProductKernel,
        RBFKernel as RBFKernel,
        SharedIndependentKernel as SharedIndependentKernel,
        SumKernel as SumKernel,
        WhiteNoiseKernel as WhiteNoiseKernel,
        ZeroKernel as ZeroKernel,
        cross_gram as cross_gram,
        gram as gram,
    )
    from .discrepancy_models import (
        AbstractDiscrepancyModel as AbstractDiscrepancyModel,
        GaussianProcess as GaussianProcess,
    )
    from .distributions import (
        AbstractDistribution as AbstractDistribution,
        CenteredUniform as CenteredUniform,
        Gamma as Gamma,
        Joint as Joint,
        LogNormal as LogNormal,
        Normal as Normal,
        RelativeNormal as RelativeNormal,
        RelativeTruncatedNormal as RelativeTruncatedNormal,
        Transformed as Transformed,
        TruncatedNormal as TruncatedNormal,
        Uniform as Uniform,
        truncate as truncate,
    )
    from .likelihoods import (
        AbstractLikelihood as AbstractLikelihood,
        GaussianLikelihood as GaussianLikelihood,
    )
    from .linearization import (
        Linearization as Linearization,
        posterior_covariance as posterior_covariance,
    )
    from .noise_models import (
        AbstractNoiseModel as AbstractNoiseModel,
        AutoCrossNoise as AutoCrossNoise,
    )

_EXPORTS = {
    'AbstractBijector': 'bijectors',
    'AbstractCovarianceKernel': 'covariance_kernels',
    'AbstractDiscrepancyModel': 'discrepancy_models',
    'AbstractDistribution': 'distributions',
    'AbstractLikelihood': 'likelihoods',
    'AbstractNoiseModel': 'noise_models',
    'AutoCrossKernel': 'covariance_kernels',
    'AutoCrossNoise': 'noise_models',
    'CenteredUniform': 'distributions',
    'Chain': 'bijectors',
    'ConstantKernel': 'covariance_kernels',
    'CosineKernel': 'covariance_kernels',
    'DiagLinear': 'bijectors',
    'Exp': 'bijectors',
    'Gamma': 'distributions',
    'GaussianLikelihood': 'likelihoods',
    'GaussianProcess': 'discrepancy_models',
    'Identity': 'bijectors',
    'Inverse': 'bijectors',
    'Joint': 'distributions',
    'Leafwise': 'bijectors',
    'Linearization': 'linearization',
    'LogNormal': 'distributions',
    'Matern32Kernel': 'covariance_kernels',
    'Matern52Kernel': 'covariance_kernels',
    'Normal': 'distributions',
    'PeriodicKernel': 'covariance_kernels',
    'Permute': 'bijectors',
    'ProductKernel': 'covariance_kernels',
    'R2ToComplex': 'bijectors',
    'RBFKernel': 'covariance_kernels',
    'RelativeNormal': 'distributions',
    'RelativeTruncatedNormal': 'distributions',
    'ScalarAffine': 'bijectors',
    'SharedIndependentKernel': 'covariance_kernels',
    'Shift': 'bijectors',
    'Sigmoid': 'bijectors',
    'Softplus': 'bijectors',
    'SumKernel': 'covariance_kernels',
    'Tanh': 'bijectors',
    'Transformed': 'distributions',
    'Transpose': 'bijectors',
    'TriangularLinear': 'bijectors',
    'TruncatedNormal': 'distributions',
    'Uniform': 'distributions',
    'WhiteNoiseKernel': 'covariance_kernels',
    'ZeroKernel': 'covariance_kernels',
    'cross_gram': 'covariance_kernels',
    'gram': 'covariance_kernels',
    'posterior_covariance': 'linearization',
    'truncate': 'distributions',
}
_MODULES = {'bijectors', 'discrepancy_models', 'linearization', 'likelihoods', 'noise_models', 'distributions', 'covariance_kernels'}

def __getattr__(name):
    if name in _MODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _EXPORTS:
        value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(set(globals()) | _MODULES | _EXPORTS.keys())

__all__ = sorted(_MODULES | _EXPORTS.keys())
