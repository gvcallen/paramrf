# from pmrf.core.network import Network
# from pmrf.core.parameter import Parameter

from pmrf._model import (
    Model as Model,
)

from pmrf._system import (
    ModelSystem as ModelSystem,
)

from pmrf._parameter import (
    Parameter as Parameter,
    field as field,
)

from pmrf._numpy import (
    numpy as numpy
)

from importlib.metadata import version as _version, PackageNotFoundError
try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass