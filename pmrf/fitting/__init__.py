import inspect

from pmrf.fitting.base import (
    Fitter,
    INIT_PARAMS,
    BaseFitter,
    FitResults,
    FitContext,
    is_frequentist,
    is_bayesian,
    is_inference_kind,
)

from pmrf.fitting.frequentist import (
    FrequentistFitter,
    FrequentistResults,
    FrequentistContext,
)

from pmrf.fitting.bayesian import (
    BayesianFitter,
    BayesianResults,
    BayesianContext,
)

from pmrf.fitting._backends import *

fitter_classes = [BaseFitter, FrequentistFitter, BayesianFitter]
FITTER_INIT_PARAMS = []

for cls in fitter_classes:
    # Get the signature of the __init__ method
    sig = inspect.signature(cls.__init__)
    
    # Extract parameter names, filtering out 'self' and *args/**kwargs
    params = [
        param.name 
        for param in sig.parameters.values() 
        if param.name != 'self' 
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    
    FITTER_INIT_PARAMS.extend(params)
    
FITTER_INIT_PARAMS = list(set(FITTER_INIT_PARAMS))
FITTER_INIT_PARAMS.remove('model')
FITTER_INIT_PARAMS.extend(['inference', 'backend'])