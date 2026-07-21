"""
Resolution of measured data into the datasets a fit is composed of.
"""

from typing import Callable, NamedTuple, Sequence

import numpy as np
import jax.numpy as jnp
try:
    import skrf
except ImportError:
    pass

from pmrf.evaluators import Feature
from pmrf.frequency import Frequency
from pmrf.models import SkrfNetwork
from pmrf.network_collection import NetworkCollection


class Dataset(NamedTuple):
    """One measured dataset, and the model-side predictor and grid it is fitted on."""
    #: The evaluator extracting the fitted features from the model.
    predictor: Callable[..., jnp.ndarray]

    #: The measured values the predictor is compared against.
    target: np.ndarray

    #: The frequency axis this dataset was measured on.
    frequency: Frequency


def resolve_datasets(
    features: str | Sequence[str] | Callable,
    data,
    frequency: Frequency | None = None,
) -> tuple[Dataset, ...]:
    """
    Resolve measured data into the datasets to be fitted.

    A :class:`pmrf.NetworkCollection` yields one dataset per network, each retaining
    its own native frequency axis and each predicted by the correspondingly named
    sub-model. Any other data yields a single dataset.

    Parameters
    ----------
    features : str | Sequence[str] | Callable
        The RF features to fit. A callable is used as the predictor directly, in
        which case a collection cannot be split up and one dataset is returned.
    data : np.ndarray | jnp.ndarray | skrf.Network | NetworkCollection
        The data to fit to.
    frequency : Frequency | None, default=None
        The frequency sweep. Required for raw array data, and otherwise taken from
        the network(s).

    Returns
    -------
    tuple[Dataset, ...]
        The resolved datasets.
    """
    if isinstance(features, Callable):
        # An opaque predictor cannot be re-pointed at an individual network, so the
        # data is left whole and the caller's frequency governs.
        if isinstance(data, skrf.Network) and frequency is None:
            frequency = Frequency.from_skrf(data.frequency)
        target = data if not isinstance(data, skrf.Network) else features(SkrfNetwork(data), frequency)
        return (Dataset(features, target, frequency),)

    if isinstance(data, NetworkCollection):
        names = [ntwk.name for ntwk in data]
        if len(names) != len(set(names)):
            raise ValueError(
                "Multiple networks with the same name found in `data`. "
                "Names must be unique so each can be matched to its sub-model."
            )

        datasets = []
        for ntwk in data:
            ntwk_frequency = frequency if frequency is not None else Frequency.from_skrf(ntwk.frequency)
            target = Feature(features)(SkrfNetwork(ntwk), ntwk_frequency)
            datasets.append(Dataset(Feature(_prefix(features, ntwk.name)), target, ntwk_frequency))
        return tuple(datasets)

    predictor = Feature(features)
    if isinstance(data, skrf.Network):
        if frequency is None:
            frequency = Frequency.from_skrf(data.frequency)
        return (Dataset(predictor, predictor(SkrfNetwork(data), frequency), frequency),)

    return (Dataset(predictor, data, frequency),)


def union_frequency(datasets: tuple[Dataset, ...]) -> Frequency | None:
    """
    The frequency axis spanning every dataset, used for reporting and plotting.

    Returns the axis itself when the datasets already share one, so that a
    homogeneous fit reports exactly the grid it was fitted on.
    """
    grids = [d.frequency for d in datasets if d.frequency is not None]
    if not grids:
        return None
    if len(grids) == 1:
        return grids[0]

    f = np.unique(np.concatenate([np.asarray(g.f) for g in grids]))
    return Frequency.from_f(f / grids[0].multiplier, unit=grids[0].unit)


def _prefix(features: str | Sequence[str], name: str) -> str | list[str]:
    if isinstance(features, str):
        return f"{name}.{features}"
    return [f"{name}.{feature}" for feature in features]
