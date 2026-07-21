"""
Callables that evaluate a model, which form the summands of a problem.
"""

from typing import Callable, Sequence, TypeAlias
from abc import abstractmethod

import jax.numpy as jnp
import equinox as eqx
import eqxpress as ex
import parax as prx

from pmrf.models.base import Model
from pmrf.frequency import Frequency
from pmrf.evaluators import AbstractEvaluator
from pmrf.utils import freeze, field, unwrap
from pmrf.utils.tree import log_prob


class AbstractTerm(eqx.Module):
    """
    Abstract base class for callables that evaluate a model.

    Terms are the summands of a :class:`pmrf.Problem`. An evaluator takes both a model
    and a frequency, whereas a term takes only a model, having already resolved the
    frequency that it needs.

    Inheriting from this class is optional. Any callable that accepts a model is a
    valid term, meaning that a once-off penalty can simply be written as a function.
    """
    @abstractmethod
    def __call__(self, model: Model, **kwargs) -> jnp.ndarray:
        """
        Evaluate the model.

        Parameters
        ----------
        model : Model
            The model instance to evaluate.
        **kwargs : dict
            Additional keyword arguments for the evaluation process.

        Returns
        -------
        jnp.ndarray
            The evaluated result.
        """
        raise NotImplementedError


#: A type alias for the function signature of a term.
TermFn: TypeAlias = Callable[[Model], jnp.ndarray]

#: A type alias for "term-like" objects, used as inputs to functions.
TermLike: TypeAlias = TermFn | tuple[Callable, Frequency]


class BoundEvaluator(AbstractTerm):
    """
    Binds an evaluator to the frequency sweep it should be evaluated over.

    Evaluators take frequency as input, meaning the same evaluator can be re-used over
    different sweeps. Binding one to a particular sweep creates a term, which can then
    be summed by a :class:`pmrf.Problem`. Several evaluators can therefore be bound to
    different sweeps and solved together, allowing a single set of model parameters to
    be fitted over multiple bands at once.

    Parameters
    ----------
    evaluator
        The operator (e.g. a Loss or Likelihood) to evaluate.
    frequency
        The frequency range or points to evaluate the operator over.
    weight
        A scaling factor applied to this term's contribution to the total.
    """
    #: The active evaluator.
    evaluator: AbstractEvaluator

    #: The frequency domain this evaluator is bound to.
    frequency: Frequency = field(converter=freeze)

    #: The relative weight of this term.
    weight: float = field(default=1.0, static=True)

    def __call__(self, model: Model, **kwargs) -> jnp.ndarray:
        return self.weight * unwrap(self.evaluator)(model, unwrap(self.frequency), **kwargs)


class NegativeLogPrior(AbstractTerm):
    """
    Computes the negative of the log of the prior on a model's parameters.

    This is a term and not an evaluator, since it depends only on the parameters and
    therefore has no frequency sweep to be evaluated over. When summed alongside a
    likelihood for each dataset, it penalizes the shared parameters once, as opposed
    to once per dataset.

    The prior is assumed to be attached to the model and the log prior probability
    is evaluated over the model's current parameters. This can be done either by
    specifying the distributions of individual parameters using
    :class:`pmrf.parameters.Random`, or by attaching joint probability distributions
    using :class:`pmrf.models.Probabilistic`.
    """
    def __call__(self, model: Model, **kwargs) -> jnp.ndarray:
        return -log_prob(model)


def as_terms(objective: TermLike | Sequence[TermLike], frequency: Frequency | None = None) -> tuple[TermFn, ...]:
    """
    Normalize a user-supplied objective into a tuple of terms.

    Accepts either a single objective or a sequence of them. Each element can be an
    evaluator, an ``(evaluator, frequency)`` pair binding it to its own sweep, an
    existing term, or a plain callable that accepts only the model.

    Parameters
    ----------
    objective
        The objective(s) to normalize.
    frequency
        The frequency used for any element that does not carry its own. Only required
        if at least one element is not already bound to a sweep.

    Returns
    -------
    tuple
        The resolved terms.
    """
    elements = objective if isinstance(objective, (list, tuple)) and not _is_pair(objective) else [objective]
    return tuple(_as_term(element, frequency) for element in elements)


def _is_pair(obj) -> bool:
    return len(obj) == 2 and callable(obj[0]) and isinstance(obj[1], Frequency)


def _as_term(element: TermLike, frequency: Frequency | None) -> TermFn:
    if isinstance(element, (list, tuple)):
        if not _is_pair(element):
            raise TypeError(
                "A paired objective must be an (evaluator, frequency) tuple. "
                f"Got a {type(element).__name__} of length {len(element)}."
            )
        return BoundEvaluator(_as_evaluator(element[0]), element[1])

    if isinstance(element, AbstractTerm):
        return element

    if frequency is None:
        # Nothing to bind to, so the element must already accept the model alone.
        return _as_evaluator(element)

    return BoundEvaluator(_as_evaluator(element), frequency)


def _as_evaluator(fn: Callable) -> Callable:
    return fn if isinstance(fn, eqx.Module) else prx.Static(ex.Lambda(fn))


__all__ = [
    'AbstractTerm',
    'BoundEvaluator',
    'NegativeLogPrior',
    'TermFn',
    'TermLike',
    'as_terms',
]
