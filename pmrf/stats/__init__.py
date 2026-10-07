"""Public stats modules and their common entry points."""

from importlib import import_module

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
