"""
(experimental) model exploration via random sampling.

Parameter space exploration and active learning module.

This module provides engines for non-adaptive (One-Shot) and adaptive 
(Active Learning) sampling of the model's parameter space. 

Exploration is performed by instantiating an engine (e.g., `LatinHypercube` 
or `EqxLearnUncertainty`) and passing it to the unified `sample` function 
along with the model. The router automatically delegates to the correct 
execution loop, evaluating the physical parameters and extracted features.

Results are returned as an `ExploreResult`, which contains the
batched models and raw arrays for downstream plotting or surrogate training.
"""

from pmrf.explore.sample import sample
from pmrf.explore.result import ExploreResult
from pmrf.explore.samplers import (
    AbstractSampler,
    AbstractOneShotSampler,
    AbstractAdaptiveSampler,
    LatinHypercube,
    Uniform,
    FieldSampler,
    EqxLearnUncertainty
)

__all__ = [
    "sample",
    "ExploreResult",
    "AbstractSampler",
    "AbstractOneShotSampler",
    "AbstractAdaptiveSampler",
    "LatinHypercube",
    "Uniform",
    "FieldSampler",
    "EqxLearnUncertainty",
]