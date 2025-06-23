from pmrf._model import (
    Model as Model,
    model_check as model_check
)
from pmrf.parameters import *

from pmrf._frequency import (
    Frequency as Frequency,
)

from pmrf._misc import (
    field,
)

from pmrf._tree import (
    partition,
    combine,
)


from importlib.metadata import version as _version, PackageNotFoundError
try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass