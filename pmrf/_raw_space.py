"""
The raw-space view of a model, shared by the minimiser and every sampler.

Solvers move through a model's free parameters as a name-keyed dict of raw values.
Everything here goes through the public :func:`pmrf.param_values`,
:func:`pmrf.update` and :func:`pmrf.log_prior`, so a solver sees exactly what a user
would with those functions.
"""

from typing import Any, Callable

from jaxtyping import PyTree, Scalar

from pmrf.parameters import log_prior, param_values, update
from pmrf.utils import unwrap


class RawSpace:
    """
    The free parameters of a model in raw space.

    Parameters
    ----------
    model : PyTree
        A model, or any collection of models and parameters.
    action : str
        What the solver does, such as ``'optimize'``, for the error raised when there
        is nothing to move.

    Raises
    ------
    ValueError
        If `model` has no free parameters.
    """

    def __init__(self, model: PyTree, action: str):
        #: The model the raw values are written into.
        self.model = model
        #: The starting raw values of the free parameters, by name.
        self.y0 = param_values(model, free_only=True, space='raw')
        if not self.y0:
            raise ValueError(
                f"Nothing to {action}: the tree has no free parameters. Every parameter is "
                "either fixed or a plain value."
            )

    def updated(self, values: dict) -> PyTree:
        """Returns the model with `values`, raw and by name, written into it.

        Batched values give a batched model; fixed parameters stay unbatched.
        """
        return update(self.model, values, space='raw')

    def read(self, batch: PyTree) -> dict:
        """Returns the raw values of the free parameters of `batch`, a model like this one."""
        values = param_values(batch, free_only=True, space='raw')
        return {name: values[name] for name in self.y0}

    def objective(self, fn: Callable[[PyTree, Any], Scalar]) -> Callable[[dict, Any], Scalar]:
        """Returns `fn`, which takes the unwrapped model, as a function of raw values."""
        def raw_fn(values: dict, args: Any) -> Scalar:
            return fn(unwrap(self.updated(values)), args)
        return raw_fn

    def log_prior(self, values: dict, _args: Any = None) -> Scalar:
        """Returns the raw-space log prior of the model at `values`."""
        return log_prior(self.updated(values), space='raw')

    def log_posterior(self, loglikelihood_fn: Callable[[PyTree, Any], Scalar]) -> Callable[[dict, Any], Scalar]:
        """Returns the raw-space log posterior: `loglikelihood_fn` plus the raw log prior."""
        def raw_fn(values: dict, args: Any) -> Scalar:
            model = self.updated(values)
            return loglikelihood_fn(unwrap(model), args) + log_prior(model, space='raw')
        return raw_fn
