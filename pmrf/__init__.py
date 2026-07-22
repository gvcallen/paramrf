import logging
import warnings
import os

# Stop thread contention between vmap and CPU backend multithreading
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    from threadpoolctl import threadpool_limits
    # Dynamically clamp any running BLAS/OpenMP libraries to 1 thread
    threadpool_limits(limits=1, user_api='blas')
    threadpool_limits(limits=1, user_api='openmp')
except ImportError:
    warnings.warn(
        "The 'threadpoolctl' package is not installed. If you experience "
        "hanging or CPU thrashing during circuit evaluations, ensure you "
        "import pmrf BEFORE importing jax or numpy, or install threadpoolctl"
    )

# Supress the JAX gpu warning
class _SuppressJaxGpuWarning(logging.Filter):
    def filter(self, record):
        return "An NVIDIA GPU may be present" not in record.getMessage()

jax_logger = logging.getLogger("jax._src.xla_bridge")
jax_logger.addFilter(_SuppressJaxGpuWarning())

import logging
import jax
from importlib.metadata import version as _version, PackageNotFoundError
from jaxtyping import ArrayLike
from typing import TypeAlias

# Environment Setup
jax.config.update("jax_enable_x64", True)

# Versioning
try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

# Re-exports
from pmrf.models import (
    Model as Model,
    is_model as is_model,
)
from pmrf.frequency import Frequency as Frequency

#: The canonical type hint for a float, or a numpy or JAX array.
ArrayLike: TypeAlias = ArrayLike

from pmrf.parameters import (
    Param as Param,
    is_param as is_param,
    as_param as as_param,
    param as param,
    Unconstrained as Unconstrained,
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
    Attrgetter as Attrgetter,
    Pathgetter as Pathgetter,
    field as field,
    partition as partition,
    combine as combine,
    batch_axes as batch_axes,
    batch_mask as batch_mask,
    freeze as freeze,
    unfreeze as unfreeze,
    replace as replace,
    unwrap as unwrap,
    unwrap_self as unwrap_self,
    derivative as derivative,
    sweep as sweep,
    is_constant as is_constant,
)

# Modules
from pmrf import (
    bijectors as bijectors,
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
    # Base/Core
    "Model",
    "is_model",
    "Frequency",
    "Topology",

    # Parameters
    "Param",
    "is_param",
    "as_param",
    "param",
    "Unconstrained",
    "Fixed",
    "Bounded",
    "Constrained",
    "Random",
    
    # Serialization
    "load",
    "save",

    # Utilities
    "Partial",
    "Attrgetter",
    "Pathgetter",
    "Initvar",
    "NetworkCollection",
    "field",
    "freeze",
    "unfreeze",
    "replace",
    "unwrap",
    "unwrap_self",
    "derivative",
    "sweep",
    
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