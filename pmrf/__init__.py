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

__all__ = []

# 3. Main API Hoisting
from pmrf.io import *
from pmrf.core import *
from pmrf import core, io

# Synchronize __all__ and apply branding
__all__.extend(core.__all__)
__all__.extend(io.__all__)

for name in core.__all__ + io.__all__:
    obj = globals().get(name)
    if hasattr(obj, "__module__"):
        obj.__module__ = "pmrf"

# 4. Sub-Modules
from pmrf import (
    constants, distributions, evaluators, infer, 
    math_functions, models, optimize, parameters, rf_functions,
    transforms,
)
from pmrf.network_collection import NetworkCollection

__all__.extend([
    "core", "io", "constants", "distributions", "evaluators", 
    "infer", "math_functions", "models", "optimize", 
    "parameters", "rf_functions", "transforms",
    "NetworkCollection",    
])