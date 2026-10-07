"""Public objectives modules and their common entry points."""

from importlib import import_module

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
