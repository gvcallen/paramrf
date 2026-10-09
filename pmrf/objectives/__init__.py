"""Public objectives modules and their common entry points."""

from importlib import import_module
from typing import TYPE_CHECKING

# Keep static analyzers aware of the public names without loading them at runtime.
if TYPE_CHECKING:
    from .evaluators import (
        AbstractEvaluator as AbstractEvaluator,
        EvaluatorFn as EvaluatorFn,
        EvaluatorLike as EvaluatorLike,
        Feature as Feature,
        GibbsMarginalLogLikelihood as GibbsMarginalLogLikelihood,
        Goal as Goal,
        MarginalLogLikelihood as MarginalLogLikelihood,
        Negated as Negated,
        TargetLoss as TargetLoss,
    )
    from .losses import (
        AbstractLoss as AbstractLoss,
        HingeLoss as HingeLoss,
        HuberLoss as HuberLoss,
        LogMSELoss as LogMSELoss,
        MSELoss as MSELoss,
        MAPELoss as MAPELoss,
        RMSELoss as RMSELoss,
    )
    from .problems import (
        AbstractProblem as AbstractProblem,
        PriorPenalized as PriorPenalized,
        SummedTerms as SummedTerms,
        problem_terms as problem_terms,
    )
    from .terms import (
        AbstractTerm as AbstractTerm,
        BoundEvaluator as BoundEvaluator,
        TermFn as TermFn,
        TermLike as TermLike,
        as_terms as as_terms,
    )

_EXPORTS = {
    'AbstractEvaluator': 'evaluators',
    'AbstractLoss': 'losses',
    'AbstractProblem': 'problems',
    'AbstractTerm': 'terms',
    'BoundEvaluator': 'terms',
    'EvaluatorFn': 'evaluators',
    'EvaluatorLike': 'evaluators',
    'Feature': 'evaluators',
    'GibbsMarginalLogLikelihood': 'evaluators',
    'Goal': 'evaluators',
    'HingeLoss': 'losses',
    'HuberLoss': 'losses',
    'LogMSELoss': 'losses',
    'MSELoss': 'losses',
    'MAPELoss': 'losses',
    'MarginalLogLikelihood': 'evaluators',
    'Negated': 'evaluators',
    'PriorPenalized': 'problems',
    'RMSELoss': 'losses',
    'SummedTerms': 'problems',
    'TargetLoss': 'evaluators',
    'TermFn': 'terms',
    'TermLike': 'terms',
    'as_terms': 'terms',
    'problem_terms': 'problems',
}
_MODULES = {'evaluators', 'losses', 'problems', 'terms'}

def __getattr__(name):
    if name in _MODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _EXPORTS:
        value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(set(globals()) | _MODULES | _EXPORTS.keys())

__all__ = sorted(_MODULES | _EXPORTS.keys())
