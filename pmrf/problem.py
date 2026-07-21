"""
A callable to be "solved" (i.e. minimized or sampled).
"""

import jax.numpy as jnp
import equinox as eqx

from pmrf.models.base import Model
from pmrf.terms import TermFn
from pmrf.utils import field, unwrap


class Problem(eqx.Module):
    """
    A callable to be "solved" (i.e. minimized or sampled).

    This class encapsulates a model and the terms to be summed over it into a single
    callable unit. Each term is a callable that accepts the model and returns a
    result. :class:`pmrf.BoundEvaluator` can be used to bind an evaluator (such as a
    loss or likelihood) to a frequency sweep, whilst a term that requires no
    frequency, such as a penalty on the parameters, can be passed as a plain
    callable.

    Parameters
    ----------
    model
        The RF model to be evaluated.
    terms
        The terms to sum. Each maps the model to a scalar or array result.
    """
    #: The active RF model.
    model: Model

    #: The terms summed to produce the result.
    terms: tuple[TermFn, ...] = field(converter=tuple)

    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        """
        Sum every term over the model.

        Returns
        -------
        jnp.ndarray
            The total across all terms.
        """
        model = unwrap(self.model)
        results = [unwrap(term)(model, *args, **kwargs) for term in self.terms]
        return sum(results[1:], start=results[0])
