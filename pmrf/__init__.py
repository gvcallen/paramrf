import logging
import jax
from importlib.metadata import version as _version, PackageNotFoundError

# Environment Setup
jax_logger = logging.getLogger("jax._src.xla_bridge")
jax_logger.setLevel(logging.ERROR)
jax.config.update("jax_enable_x64", True)

# Versioning
try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

# Re-exports
from pmrf.problem import Problem as Problem
from pmrf.models import Model as Model
from pmrf.frequency import Frequency as Frequency
from pmrf.parameters import Param as Param, param as param
from pmrf.serialization import (
    load as load,
    save as save,
)
from pmrf.jax_utils import (
    Partial as Partial,
    Tied as Tied,
    InitVar as InitVar,
    replace as replace,
    combine as combine,
    field as field,
    unwrap as unwrap,
    unwrap_self as unwrap_self,
    as_fixed as as_fixed,
    as_free as as_free,
    as_frozen as as_frozen,
)
from pmrf.network_collection import NetworkCollection as NetworkCollection

# Modules
from pmrf import (
    constraints as constraints,
    covariance_kernels as covariance_kernels,
    discrepancy_models as discrepancy_models,
    distributions as distributions,
    evaluators as evaluators,
    fitting as fitting,
    infer as infer,
    likelihoods as likelihoods,
    losses as losses,
    math as math,
    models as models,
    noise_models as noise_models,
    optimize as optimize,
    parameters as parameters,
    rf as rf,
    serialization as serialization,
    viz as viz,
)


__all__ = [
    # Core
    "Model",
    "Frequency",
    "Param",
    "param",
    
    # Utilities
    "load",
    "save",
    "combine",
    "field",
    "unwrap",
    "Partial",
    "as_fixed",
    "as_free",
    "as_frozen",
    "NetworkCollection",
    
    # Sub-modules
    "constraints",
    "covariance_kernels",
    "discrepancy_models",
    "distributions",
    "evaluators",
    "fitting",
    "infer",
    "likelihoods",
    "losses",
    "math",
    "models",
    "noise_models",
    "optimize",
    "parameters",
    "rf",
    "serialization",
    "viz",
]