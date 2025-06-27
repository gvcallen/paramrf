from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

import importlib
import logging

import skrf

from pmrf._model import Model
from pmrf._frequency import Frequency
from pmrf._constants import FeatureT, FeatureListT
from pmrf.fitting._features import extract_features, create_stacked_features

@dataclass
class FitResults:
    model: Model | None = None
    engine_results: Any = None

def Fitter(
    engine: str,
    *args,
    **kwargs
) -> 'BaseFitter':
    """Fitter factory function.
    
    This allows the creator of a fitter by simply specifying the engine and having all arguments forwarded.
    See the relevant fitter classes for detailed documentation.

    Args:
        engine (str): The engine to use, specified as either e.g. 'ScipyMinimize' or 'scipy-minimize'.

    Returns:
        BaseFitter: The concrete fitter instance.
    """
    cls = get_fitter_class(engine)
    return cls(*args, **kwargs)

class BaseFitter(ABC):
    """
    **Overview**

    An abstract base class that provides the foundational structure for all
    fitting algorithms in `pmrf`.

    This class handles the common setup tasks required for any fitting routine, including:
    - Managing the parametric `Model` to be optimized.
    - Processing and aligning the measured `skrf.Network` data.
    - Interpolating all data onto a common frequency axis.
    - Defining the logic for feature extraction, which transforms raw S-parameters
      into a format suitable for comparison (e.g., magnitude, dB, phase).
    """
    def __init__(
        self,
        model: Model,
        measured: skrf.Network | list[skrf.Network],
        frequency: skrf.Frequency | None = None,
        features: FeatureT | FeatureListT | None = None,
        dont_stack_features: bool = False,
    ) -> None:
        """Initializes the BaseFitter.

        Args:
            model (Model):                                          The parametric `pmrf` model to be fitted.
            measured (skrf.Network | list[skrf.Network]):           The measured network data to fit the model against. If a list of
                                                                    networks is passed, they are treated as a single stacked N-port network.
            frequency (skrf.Frequency | None, optional):            The frequency axis to perform the fit on. If `None`, the frequency
                                                                    from the first measured network is used. All networks will be
                                                                    interpolated onto this single frequency axis. Defaults to `None`.
            features (FeatureT | FeatureListT | None, optional):    Defines the features to be extracted from the network data for comparison.
                                                                    This can be a list of strings (e.g., `['s11_db', 's21_deg']`) to extract
                                                                    those features for all ports, or a list of (feature, ports) tuples.                                                                
                                                                    See `extract_features` for more info.
                                                                    Defaults to `None`, which uses S11 magnitude (`('s', (0, 0))`).
            dont_stack_features (bool): False                       Specifies that features should not be stacked using `create_stacked_features(..)`,
                                                                    such e.g. each network's 's11' is extracted if only ['s11'] is passed.
                                                                    Only applies in the case of a list of measured data. Defaults to False.
        """
        features = features if features is not None else 's11'
        if isinstance(measured, list) and not dont_stack_features:
            features = create_stacked_features(features, measured)
        
        # All frequencies must be the same across all measurements (at least currently..)
        measured = [measured] if not isinstance(measured, list) else measured
        if frequency is not None:
            measured = [ntwk.interpolate(frequency) for ntwk in measured]
            measured_freq = frequency
        else:
            measured_freq = measured[0].frequency
            for ntwk in measured:
                if ntwk.frequency != measured_freq and not len(ntwk.frequency) == 0:
                    raise ValueError("Error: Currently `fit_frequency` must be passed for multi-measurement fits (i.e. all networks must be explicitly interpolated onto the same frequency for fitting)")
                
        # Initialize model parameters from user and store in flat array
        self.model: Model = model
        self.model_frequency = Frequency.from_skrf(measured_freq)
        self.measured: list[skrf.Network] = measured
        self.measured_frequency = measured_freq
        self.measured_features = extract_features(measured, features)
        self.feature_list = features
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def run(self, *args, **kwargs) -> FitResults:
        """Executes the fitting algorithm.

        This method must be implemented by all concrete subclasses. It is the
        main entry point to start the optimization or sampling process.

        Returns:
            FitResults: An object containing the results of the fit.
        """
        pass    
     
def is_frequentist(engine) -> bool:
    from pmrf.fitting._frequentist import FrequentistFitter
    cls = get_fitter_class(engine)
    return issubclass(cls, FrequentistFitter)

def is_bayesian(engine) -> bool:
    from pmrf.fitting._bayesian import BayesianFitter
    cls = get_fitter_class(engine)
    return issubclass(cls, BayesianFitter)

def get_fitter_class(engine: str):
    class_name = ''.join(part.capitalize() for part in engine.split('-'))
    class_name = class_name + 'Fitter'
    try:
        frequentist = importlib.import_module('pmrf.fitting._frequentist')
        bayesian = importlib.import_module('pmrf.fitting._bayesian')
        if hasattr(frequentist, class_name):
            return getattr(frequentist, class_name)
        return getattr(bayesian, class_name)
    except (ImportError, AttributeError):
        return None
    return globals().get(class_name)