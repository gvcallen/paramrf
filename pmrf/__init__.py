import logging

# 1. Mute the logger
jax_logger = logging.getLogger("jax._src.xla_bridge")
initial_level = jax_logger.getEffectiveLevel()
jax_logger.setLevel(logging.ERROR)

import jax
jax.config.update("jax_enable_x64", True)

from pmrf.model import Model
from pmrf.frequency import Frequency
from pmrf.parameters import Parameter
from pmrf.features import Extractor
from pmrf.field import field
from pmrf.io import save, load
from pmrf.network_collection import NetworkCollection

from pmrf.algorithms import *
from pmrf.constants import *
from pmrf.distributions import *
from pmrf.fit import *
from pmrf.math_functions import *
from pmrf.models import *
from pmrf.parameters import *
from pmrf.rf_functions import *
from pmrf.sample import *
from pmrf.utils import *

from pmrf import algorithms, constants, distributions, fitting, math_functions, models, parameters, rf_functions, sampling, util

from importlib.metadata import version as _version, PackageNotFoundError
try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

__all__ = [
    "Model",
    "Parameter",
    "Frequency",
    "extract_features",
    "save",
    "load",
    "NetworkCollection",
    "BaseResults",
    "BaseRunner",
    "field",
]

__all__.extend(algorithms.__all__)
__all__.extend(constants.__all__)
__all__.extend(distributions.__all__)
__all__.extend(fitting.__all__)
__all__.extend(math_functions.__all__)
__all__.extend(models.__all__)
__all__.extend(parameters.__all__)
__all__.extend(rf_functions.__all__)
__all__.extend(sampling.__all__)
__all__.extend(util.__all__)