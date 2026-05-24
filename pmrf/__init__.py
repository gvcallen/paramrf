import logging

# Supress the JAX gpu warning
class _SuppressJaxGpuWarning(logging.Filter):
    def filter(self, record):
        return "An NVIDIA GPU may be present" not in record.getMessage()

jax_logger = logging.getLogger("jax._src.xla_bridge")
jax_logger.addFilter(_SuppressJaxGpuWarning())

import logging
import jax
from importlib.metadata import version as _version, PackageNotFoundError
import parax 
from jaxtyping import Inexact, Array
from typing import TypeAlias

# Environment Setup
jax.config.update("jax_enable_x64", True)

# Versioning
try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

# Re-exports
from pmrf.models import Model as Model
from pmrf.frequency import Frequency as Frequency
from pmrf.topology import Topology as Topology

#: The canonical type hint for a parameter in a model.
#: Parameters should be created using factories in :mod:`pmrf.parameters`,
#: most of which are re-exported at root (e.g. :func:`pmrf.Variable`, :func:`pmrf.Fixed`, :func:`pmrf.Bounded`).
Param: TypeAlias = parax.AbstractVariable | Inexact[Array, "..."]
from pmrf.parameters import (
    param as param,
    as_param as as_param,
    Variable as Variable,
    Fixed as Fixed,
    Bounded as Bounded,
    Constrained as Constrained,
    Random as Random,
)

from pmrf.serialization import (
    load as load,
    save as save,
)

from pmrf.network_collection import NetworkCollection as NetworkCollection

from pmrf.utils import (
    Partial as Partial,
    InitVar as InitVar,
    field as field,
    freeze as freeze,
    unfreeze as unfreeze,
    replace as replace,
    unwrap as unwrap,
    unwrap_self as unwrap_self,
    is_constant as is_constant,
    is_param as is_param,
    is_model as is_model,
)

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
    "as_param",
    "Variable",
    "Fixed",
    "Bounded",
    "Constrained",
    "Random",
    
    # Serialization
    "load",
    "save",

    # Utilities
    "Partial",
    "Initvar",
    "NetworkCollection",
    "field",
    "freeze",
    "unfreeze",
    "replace",
    "unwrap",
    "unwrap_self",
    "is_constant",
    "is_param",
    "is_model",
    
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