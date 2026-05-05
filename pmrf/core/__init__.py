"""
The core module, containing the core classes in ParamRF.

Re-exported at root.
"""

from pmrf.core.model import (
    Model as Model,
    model as model,
)
from pmrf.core.frequency import Frequency as Frequency
from pmrf.core.loss import (
    Loss as Loss,
    LossFn as LossFn,
    LossLike as LossLike,
)
from pmrf.core.likelihood import (
    Likelihood as Likelihood,
    NoiseModel as NoiseModel,
)
from pmrf.core.discrepancy import (
    DiscrepancyModel as DiscrepancyModel,
    CovarianceKernel as CovarianceKernel,
)
from pmrf.core.evaluator import (
    Evaluator as Evaluator,
    EvaluatorFn as EvaluatorFn,
    EvaluatorLike as EvaluatorLike,
)
from pmrf.core.problem import Problem as Problem

for _cls in (DiscrepancyModel, Evaluator, Frequency, Likelihood, Loss, Model, NoiseModel, CovarianceKernel, Problem):
    _cls.__module__ = "pmrf"

__all__ = [
    "DiscrepancyModel",
    "Evaluator",
    "Frequency",
    "Likelihood",
    "Loss",
    "Model",
    "model",
    "NoiseModel",
    "CovarianceKernel",
    "Problem",
]