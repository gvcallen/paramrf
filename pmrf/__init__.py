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
from pmrf.fitting import (
    fit as fit,
    fit_sequential as fit_sequential
)
from pmrf.network_collection import NetworkCollection as NetworkCollection

# 4. Sub-Modules
from pmrf import (
    discrepancy_models as discrepancy_models,
    evaluators as evaluators,
    explore as explore,
    fitting as fitting,
    likelihoods as likelihoods,
    losses as losses,
    infer as infer,
    models as models,
    math as math,
    optimize as optimize,
    rf as rf,
    serialization as serialization,
)