import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

from pmrf.features import extract_features
from pmrf.frequency import Frequency
from pmrf.io import save, load
from pmrf.network_collection import NetworkCollection
from pmrf.results import BaseResults
from pmrf.runner import BaseRunner

from pmrf.algorithms import *
from pmrf.constants import *
from pmrf.distributions import *
from pmrf.fitting import *
from pmrf.math_functions import *
from pmrf.models import *
from pmrf.parameters import *
from pmrf.rf_functions import *
from pmrf.sampling import *
from pmrf.util import *

from pmrf import algorithms, constants, distributions, fitting, math_functions, models, parameters, rf_functions, sampling, util

from importlib.metadata import version as _version, PackageNotFoundError
try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

__all__ = [
    "extract_features",
    "Frequency",
    "save",
    "load",
    "NetworkCollection",
    "BaseResults",
    "BaseRunner",
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