import skrf

from pmrf._model import Model
from pmrf._constants import FeatureT, FeatureListT

from pmrf.fitting._base import BaseFitter, FitResults

class BayesianResults(FitResults):
    pass
        
class BayesianFitter(BaseFitter):
    """
    **Overview**

    A base class for Bayesian fitting methods.

    This class extends `BaseFitter` by adding the concept of a `likelihood_fn`,
    as well as providing support for prior sampling.
    """
    def __init__(
        self,
        model: Model,
        measured: skrf.Network | list[skrf.Network],
        frequency: skrf.Frequency | None = None,
        features: FeatureT | FeatureListT | None = None,
        # TODO add likelihood function
        *args, **kwargs
    ) -> None:
        """Initializes the BayesianFitter.

        Args:
            model (Model):
                The parametric `pmrf` model to be fitted.
            measured (skrf.Network | list[skrf.Network]):
                The measured network data to fit the model against.
            frequency (skrf.Frequency | None, optional):
                The frequency axis to perform the fit on. Defaults to `None`.
            features (FeatureT | FeatureListT | None = None, optional):
                The features to extract for comparison. Defaults to `None`.
        """
        super().__init__(model=model, measured=measured, frequency=frequency, features=features, *args, **kwargs)
        
class PolychordFitter(BayesianFitter):
    """
    **Overview**

    A base class for Bayesian fitting methods.

    This class extends `BaseFitter` by adding the concept of a `likelihood_fn`,
    as well as providing support for prior sampling.
    """
    def __init__(
        self,
        model: Model,
        measured: skrf.Network | list[skrf.Network],
        frequency: skrf.Frequency | None = None,
        features: FeatureT | FeatureListT | None = None,
        # TODO add likelihood function
        *args, **kwargs
    ) -> None:
        """Initializes the BayesianFitter.

        Args:
            model (Model):
                The parametric `pmrf` model to be fitted.
            measured (skrf.Network | list[skrf.Network]):
                The measured network data to fit the model against.
            frequency (skrf.Frequency | None, optional):
                The frequency axis to perform the fit on. Defaults to `None`.
            features (FeatureT | FeatureListT | None = None, optional):
                The features to extract for comparison. Defaults to `None`.
        """
        super().__init__(model=model, measured=measured, frequency=frequency, features=features, *args, **kwargs)
        