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
from parax import bijectors
from pmrf.core import *
from pmrf.io import *
from pmrf import core
from pmrf import io

# Synchronize __all__ and apply branding
__all__.extend(core.__all__)
__all__.extend(io.__all__)

for name in core.__all__:
    obj = globals().get(name)
    if hasattr(obj, "__module__"):
        obj.__module__ = "pmrf"

# 4. Sub-Modules
from pmrf import (
    constants, distributions, evaluators, 
    math_functions, models,
    optimize, infer, fit, explore,
    rf_functions, losses,
)
from pmrf.network_collection import NetworkCollection

__all__.extend([
    "core", "constants", "distributions", "evaluators", 
    "math_functions", "models",
    "optimize", "infer", "fit", "explore",
    "rf_functions", "losses", "bijectors",
    "NetworkCollection",
])