import logging

# 1. Mute the logger
jax_logger = logging.getLogger("jax._src.xla_bridge")
initial_level = jax_logger.getEffectiveLevel()
jax_logger.setLevel(logging.ERROR)

import jax
jax.config.update("jax_enable_x64", True)

from pmrf.field import field
from pmrf.io import save, load
from pmrf.partition import partition
from pmrf.network_collection import NetworkCollection
from pmrf.core import *
from pmrf import core, constants, distributions, evaluators, infer, math_functions, models, optimize, parameters, rf_functions

from importlib.metadata import version as _version, PackageNotFoundError
try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

__all__ = [
    "field",
    "save", "load",
    "partition",
    "NetworkCollection",
    "core", "constants", "distributions", "evaluators", "infer", "math_functions", "models", "optimize", "parameters", "rf_functions",
]
__all__.extend(core.__all__)

# @classmethod
# def from_alias(cls, alias: str) -> 'Evaluator':
#     fields = alias.split('.')
#     if len(fields) > 1:
#         # FIX: Join back into a dot-separated string for attrgetter
#         subattrs = ".".join(fields[:-1]) 
#         alias = fields[-1]
#     else:
#         subattrs = None

#     match = re.match(r'^([a-zA-Z]+)(\d)?(\d)?(.*)$', alias)
#     if not match:
#         raise ValueError(f"Invalid feature alias format: '{alias}'")

#     prop_prefix = match.group(1)
#     port1 = match.group(2)
#     port2 = match.group(3)
#     prop_suffix = match.group(4)
    
#     prop = prop_prefix + prop_suffix

#     # Map 1-indexed string alias (e.g., S11) to 0-indexed port array slices
#     if port1 is not None and port2 is not None:
#         ports = (int(port1) - 1, int(port2) - 1)
#     else:
#         ports = None
    
#     # FIX: Changed `property` to `prop`
#     return cls(prop=prop, ports=ports, subattrs=subattrs)