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
    model as model,
    NoiseModel as NoiseModel,
    CovarianceKernel as CovarianceKernel,
    Problem as Problem,
)
from pmrf.parameters import (
    Parameter as Parameter,
    Param as Param,
    param as param,
    Positive as Positive,
    positive as positive,
    Bounded as Bounded,
    bounded as bounded,
    Uniform as Uniform,
    uniform as uniform,
    Normal as Normal,
    normal as normal,
    CenteredUniform as CenteredUniform,
    centered_uniform as centered_uniform,
    RelativeNormal as RelativeNormal,
    relative_normal as relative_normal,
)
from pmrf.serialization import (
    load as load,
    save as save,
)
from pmrf.field import field as field

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
    "model",
    "NoiseModel",
    "Problem",

    "Parameter",
    "Param",
    "param",
    "Positive",
    "positive",
    "Bounded",
    "bounded",
    "Uniform",
    "uniform",
    "Normal",
    "normal",
    "CenteredUniform",
    "centered_uniform",
    "RelativeNormal",
    "relative_normal",
    
    # Utilities & Functions
    "load",
    "save",
    "field",
    
    # Sub-modules
    "covariance_kernels",
    "discrepancy_models",
    "evaluators",
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