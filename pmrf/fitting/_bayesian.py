import skrf

from pmrf.parameters import Parameter, Uniform
from pmrf._model import Model
from pmrf._constants import FeatureInputT
from pmrf.fitting._base import BaseFitter, FitResults


class BayesianResults(FitResults):
    pass
    
class BayesianFitter(BaseFitter):
    """
    **Overview**

    A base class for Bayesian fitting methods.

    This class extends `BaseFitter` by adding the concept of a likelihood function,
    as well as providing support for prior sampling.
    """
    def __init__(
        self,
        model: Model,
        measured: skrf.Network | dict[str, skrf.Network],
        frequency: skrf.Frequency | None = None,
        features: FeatureInputT | None = None,
        likelihood: str | None = "gaussian",
        likelihood_params: dict[str, Parameter] = None,
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
                The features to extract for comparison.
                Note that note all features are compatibile with all likelihoods,
                but no error checking is currently done for this.
                Defaults to `None`.
        """
        if likelihood != "gaussian":
            raise Exception("Currently only a gaussian likelihood is supported")
        
        super().__init__(model=model, measured=measured, frequency=frequency, features=features, *args, **kwargs)
        
        self.likelihood_params = likelihood_params if likelihood_params is not None else {'sigma': Uniform(0.0, 50.0e-3)}
