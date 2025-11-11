from pmrf.fitting._base import (
    Fitter,
    BaseFitter,
    FitResults,
    is_frequentist,
    is_bayesian,
    is_inference_kind,
)

from pmrf.fitting._frequentist import (
    FrequentistFitter,
    FrequentistResults,
)

from pmrf.fitting._bayesian import (
    BayesianFitter,
    BayesianResults,
)

try:
    from pmrf.fitting.fitters._scipy import SciPyMinimizeFitter
except:
    pass

try:
    from pmrf.fitting.fitters._polychord import PolyChordFitter
except:
    pass

try:
    from pmrf.fitting.fitters._dypolychord import dyPolyChordFitter
except:
    pass

try:
    from pmrf.fitting.fitters._numpyro import NumPyroMCMCFitter, NumPyroNSFitter
except:
    pass

try:
    from pmrf.fitting.fitters._blackjax import BlackJAXNSFitter
except:
    pass