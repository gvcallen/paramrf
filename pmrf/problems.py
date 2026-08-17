"""
A callable to be "solved" (i.e. minimized or sampled).
"""

import warnings
from abc import abstractmethod
from typing import Any

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree

from pmrf.parameters import tree_param_distributions, tree_param_log_prob
from pmrf.terms import TermFn
from pmrf.utils import field, freeze, unwrap


class AbstractProblem(eqx.Module):
    """
    Abstract base class for callables to be "solved" (i.e. minimized or sampled).

    A problem groups the PyTrees a solve is defined over into a single PyTree, and
    binds them to the terms evaluating them. Where an evaluator maps a model and a
    frequency to an array, and a term maps a PyTree to an array, a problem takes
    nothing further.

    Problems compose: one may wrap another to add to its result, in which case it
    exposes the wrapped problem's `model` as its own.
    """
    #: The tree of parameters being solved for.
    model: eqx.AbstractVar[PyTree]

    @property
    def inner(self) -> 'AbstractProblem':
        """
        The problem itself, or the one it wraps, looking through any nesting.
        """
        return self

    @abstractmethod
    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        """
        Evaluate the problem.
        """
        raise NotImplementedError


class SummedTerms(AbstractProblem):
    """
    A problem summing a set of terms over a tree of parameters.

    Each term is a callable that accepts the tree and returns a result.
    :class:`pmrf.BoundEvaluator` can be used to bind an evaluator (such as a loss or
    likelihood) to a frequency sweep, whilst a term that requires no frequency, such
    as a penalty on the parameters, can be passed as a plain callable.

    Parameters
    ----------
    model
        The tree of parameters to evaluate, typically an RF model.
    terms
        The terms to sum. Each maps the tree to a scalar or array result.
    """
    #: The tree of parameters being solved for.
    model: PyTree

    #: The terms summed to produce the result.
    terms: tuple[TermFn, ...] = field(converter=tuple)

    def __post_init__(self):
        if not self.terms:
            raise ValueError("A problem must have at least one term to sum.")

    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        model = unwrap(self.model)
        results = [unwrap(term)(model, *args, **kwargs) for term in self.terms]
        return sum(results[1:], start=results[0])


# `distributions` is derived from the problem and must never take gradients, so it is
# set in `__post_init__` rather than passed in. Equinox warns about this in general
# because such a field is easy to leave stale; here it is rebuilt from the problem it
# accompanies and read only during evaluation.
warnings.filterwarnings("ignore", message=r"Using `field\(init=False\)`")


class PriorPenalized(AbstractProblem):
    """
    A problem penalized by the negative log prior of its parameters.

    Minimizing a penalized problem gives the maximum a posteriori estimate, where the
    problem alone gives the maximum likelihood estimate. Priors are extracted over the
    whole problem, so the terms' own hyper-parameters are covered alongside the
    model's parameters.

    Priors are metadata and are stripped by unwrapping, so they are extracted once on
    construction while the problem is still wrapped.

    Parameters
    ----------
    problem
        The problem to penalize.
    """
    #: The problem being penalized.
    problem: AbstractProblem

    #: The prior distributions of the problem's parameters.
    distributions: Any = field(converter=freeze, init=False)

    def __post_init__(self):
        if isinstance(self.problem, PriorPenalized):
            raise ValueError(
                "This problem is already prior-penalized. Penalizing it again would "
                "count every prior twice."
            )
        self.distributions = tree_param_distributions(self.problem)

    @property
    def model(self) -> PyTree:
        return self.problem.model

    @property
    def inner(self) -> AbstractProblem:
        return self.problem.inner

    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        # The distributions are positioned as the parameters are once unwrapped, so the
        # values must be too. A solver will already have done this, but not a direct call.
        log_prior = tree_param_log_prob(unwrap(self.distributions), unwrap(self.problem))
        return self.problem(*args, **kwargs) - log_prior


def problem_terms(problem: AbstractProblem, name: str) -> tuple[TermFn, ...]:
    """
    Returns the terms of a problem, looking through any wrappers.

    Raises
    ------
    ValueError
        If the problem is not built from terms, in which case it should be inspected
        directly.
    """
    inner = problem.inner
    if not isinstance(inner, SummedTerms):
        raise ValueError(
            f"`{name}` is only defined for a problem built from terms, but this one is "
            f"a {type(inner).__name__}. Use `.problem` instead."
        )
    return inner.terms
