import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

from pmrf.models.model import (
    Model as Model,
    wrap,
)
from pmrf._util import (
    field,
)

from pmrf.frequency import (
    Frequency as Frequency,
)

from pmrf._features import (
    extract_features,
    _format_features,
)

from pmrf.parameters import (
    Parameter as Parameter,
    ParameterGroup as ParameterGroup,
)

from pmrf.network_collection import (
    NetworkCollection,
)

from pmrf.functions import *

from importlib.metadata import version as _version, PackageNotFoundError
try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

__all__ = [
    "Model",
    "Frequency",
    "Parameter",
    "ParameterGroup",
    "NetworkCollection",
    "wrap",
    "extract_features",
    "learn",
]