from dataclasses import dataclass
import logging

import numpy as np
import jax
import skrf

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.constants import FeatureT
from pmrf.frequency import Frequency
from pmrf import extract_features, wrap
from pmrf.network_collection import NetworkCollection
from pmrf.fitting.results import FitSettings

@dataclass
class FitContext:
    """
    Context object holding the state and data required for a fit execution.

    Attributes
    ----------
    model : Model
        The parametric model being fitted.
    measured : skrf.Network or NetworkCollection
        The target measured data.
    frequency : Frequency
        The frequency object defining the domain.
    features : list of FeatureT
        The specific features to match against.
    measured_features : np.ndarray
        The values of the features extracted from the measured data.
    output_path : str or None
        Directory path for output files.
    output_root : str or None
        Root filename for outputs.
    sparam_kind : str or None
        The S-parameter representation kind (e.g., 'all', 'transmission').
    logger : logging.Logger or None
        Logger instance for tracking progress.
    """
    model: Model
    measured: skrf.Network | NetworkCollection
    frequency: Frequency
    features: list[FeatureT]
    measured_features: np.ndarray
    output_path: str | None = None
    output_root: str | None = None
    sparam_kind: str | None = None
    logger: logging.Logger | None = None
    
    def model_param_names(self) -> list[str]:
        """
        Get the names of the flat parameters of the model.

        Returns
        -------
        list of str
            The list of parameter names.
        """
        return self.model.flat_param_names()
    
    def make_feature_function(self, as_numpy=False):
        """
        Create a JIT-compiled function to extract features from model parameters.

        Parameters
        ----------
        as_numpy : bool, default=False
            If True, the returned function handles NumPy arrays; otherwise JAX arrays.

        Returns
        -------
        callable
            A function taking ``theta`` and returning feature values.
        """
        general_feature_fn = wrap(extract_features, self.model, self.frequency, as_numpy=as_numpy)
        feature_fn = lambda theta: general_feature_fn(theta, self.features, sparam_kind=self.sparam_kind)
        return jax.jit(feature_fn)
    
    def settings(self, solver_kwargs=None, fitter_kwargs=None) -> FitSettings:
        """
        Create a FitSettings object from the current context.

        Parameters
        ----------
        solver_kwargs : dict, optional
            Solver specific arguments.
        fitter_kwargs : dict, optional
            Fitter specific arguments.

        Returns
        -------
        FitSettings
            The populated settings object.
        """
        return FitSettings(frequency=self.frequency, features=self.features, fitter_kwargs=fitter_kwargs, solver_kwargs=solver_kwargs)    