import logging
import jax
from importlib.metadata import version as _version, PackageNotFoundError

# 1. Environment Setup
jax_logger = logging.getLogger("jax._src.xla_bridge")
jax_logger.setLevel(logging.ERROR)
jax.config.update("jax_enable_x64", True)

# 2. Versioning
try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

# 3. Main API Hoisting
from pmrf.core import (
    DiscrepancyModel as DiscrepancyModel,
    Evaluator as Evaluator,
    Frequency as Frequency,
    Likelihood as Likelihood,
    Loss as Loss,
    Model as Model,
    NoiseModel as NoiseModel,
    CovarianceKernel as CovarianceKernel,
    Problem as Problem,
)
from pmrf.serialization import (
    load as load,
    save as save,
)

try:
    import skrf as rf
    skrf_available = True
    from pmrf.network_collection import NetworkCollection as NetworkCollection
except ImportError:
    skrf_available = False
    pass

# 4. Sub-Modules
from pmrf import (
    covariance_kernels as covariance_kernels,
    discrepancy_models as discrepancy_models,
    evaluators as evaluators,
    explore as explore,
    fitting as fitting,
    likelihoods as likelihoods,
    losses as losses,
    infer as infer,
    models as models,
    math as math,
    noise_models as noise_models,
    optimize as optimize,
    rf as rf,
    serialization as serialization,
    viz as viz,
)


__all__ = [
    # Core
    "CovarianceKernel",
    "DiscrepancyModel",
    "Evaluator",
    "Frequency",
    "Likelihood",
    "Loss",
    "Model",
    "NoiseModel",
    "Problem",
    
    # Utilities & Functions
    "load",
    "save",
    
    # Sub-modules
    "covariance_kernels",
    "discrepancy_models",
    "evaluators",
    "explore",
    "fitting",
    "infer",
    "likelihoods",
    "losses",
    "math",
    "models",
    "noise_models",
    "optimize",
    "rf",
    "serialization",
    "viz",
]

if skrf_available:
    __all__.append('NetworkCollection')