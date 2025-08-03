from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import Counter
import importlib
import logging
from typing import Any, Sequence
from io import BytesIO

import jax.numpy as jnp
import numpy as np
import json
import skrf
import h5py
import jsonpickle
import equinox as eqx
import skrf
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    rank = 0

from pmrf._model import Model, make_feature_fn
from pmrf._frequency import Frequency
from pmrf._constants import FeatureT
from pmrf._util import LevelFilteredLogger, iter_submodules, load_class_from_string
from pmrf._model import Model
from pmrf._frequency import Frequency
from pmrf._constants import FeatureInputT
from pmrf._features import extract_features

def Fitter(
    fitter: str,
    *args,
    **kwargs
) -> 'BaseFitter':
    """Fitter factory function.
    
    This allows the creator of a fitter by simply specifying the fitter type and having all arguments forwarded.
    See the relevant fitter classes for detailed documentation.

    Args:
        fitter (str): The fitter to use, specified as either e.g. 'ScipyMinimize' or 'scipy-minimize'.

    Returns:
        BaseFitter: The concrete fitter instance.
    """
    cls = get_fitter_class(fitter)
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
        measured: str | skrf.Network | dict[str, skrf.Network],
        frequency: skrf.Frequency | None = None,
        features: FeatureInputT | None = None,
    ) -> None:
        """Initializes the BaseFitter.

        Args:
            model (Model):                                              The parametric `pmrf` model to be fitted.
            measured (str | skrf.Network | dict[str, skrf.Network]):    The measured network data to fit the model against.
                                                                        A dict can optionally be passed, in which case
                                                                        the keys of the networks must can be referenced during
                                                                        feature extraction by also specifying features as a dictionary.
                                                                        See the documentation for the `features` argument below.
            frequency (skrf.Frequency | None, optional):                The frequency axis to perform the fit on. If `None`, the frequency
                                                                        from the first measured network is used. All networks will be
                                                                        interpolated onto this single frequency axis. Defaults to `None`.
            features (FeatureInputT | None, optional):                  Defines the features to be extracted from the network data and model for fitting.
                                                                        See `extract_features(..)` for a detail explanation. As an overview using string aliases,
                                                                        this can be a single feature e.g. 's11', a list of features (e.g., `['s11', 's11_mag']`),
                                                                        or a dictionary with either of the above as value. In the dictionary case,
                                                                        keys must be network names in the sequence passed by `measured`, which must also
                                                                        correspond to submodels which are attributes of the model. As an example,
                                                                        {'source_name1', ('s11'), {'source_name2', ('s21')} can be passed.
                                                                        Note that if a sequence of networks is passed, but a dictionary is not.
                                                                        it is assumed that those feature(s) should be extract for all measured networks/submodels.
                                                                        Defaults to `None`, which uses S11 magnitude `('s', (0, 0))`.
        """
        if isinstance(measured, str):
            measured = skrf.Network(str)
        
        # Set the default features and ensure it is not a scalar
        features = features if features is not None else 's11'
        if not isinstance(features, Sequence) and not isinstance(features, dict):
            features = [features]
        if isinstance(measured, dict) and not isinstance(features, dict):
            features = {k: features for k in measured.keys()}
        
        # All frequencies must be the same across all measurements (at least currently..). We copy the input dict
        measured = measured.copy()
        if frequency is not None:
            measured_freq = frequency
            if isinstance(measured, dict):
                measured = {k: v.interpolate(frequency) for k, v in measured}
            else:
                measured = measured.interpolate(frequency)
        else:
            measured_freq = None
            if isinstance(measured, dict):
                for ntwk in measured.values():
                    if measured_freq is None:
                        measured_freq = ntwk.frequency
                    if ntwk.frequency != measured_freq and not len(ntwk.frequency) == 0:
                        raise ValueError("Error: Currently `fit_frequency` must be passed for multi-measurement fits (i.e. all networks must be explicitly interpolated onto the same frequency for fitting)")
            else:
                measured_freq = measured.frequency
                
        # Initialize model parameters from user and store in flat array
        self.initial_model: Model = model
        self.model_frequency: Frequency = Frequency.from_skrf(measured_freq)
        self.measured: skrf.Network | dict[str, skrf.Network] = measured
        self.measured_frequency: skrf.Frequency = measured_freq
        self.measured_features = extract_features(measured, features)
        self.feature_list = features
        if rank == 0:
            self.logger = logging.getLogger("pmrf.fitting")
        else:
            self.logger = LevelFilteredLogger(null_level=logging.WARNING)        

    @abstractmethod
    def run(self, *args, **kwargs) -> 'FitResults':
        """Executes the fitting algorithm.

        This method must be implemented by all concrete subclasses. It is the
        main entry point to start the optimization or sampling process.

        Returns:
            FitResults: An object containing the results of the fit.
        """
        pass
    
    def _make_feature_function(self, as_numpy=False):
        return make_feature_fn(self.initial_model, self.feature_list, self.model_frequency, as_numpy=as_numpy)    

    
@dataclass
class FitResults:
    measured: skrf.Network | dict[str, skrf.Network] | None = None
    initial_model: Model | None = None
    fit_model: Model | None = None
    solver_results: Any = None
    frequency: Frequency | None = None
    features: list[FeatureT] | None = None
    logger: logging.Logger | None = None
    fit_args: tuple | None = None
    fit_kwargs: tuple | None = None
    solver_args: tuple | None = None
    solver_kwargs: dict | None = None
    version: int = 2
    
    def encode_solver_results(self, group: h5py.Group):
        data = None
        if self.solver_results is not None:
            try:
                data = jsonpickle.encode(self.solver_results)
            except Exception as e:
                logging.error(f"Failed to encode solver results: {e}")
        group['data'] = data
    
    @classmethod
    def decode_solver_results(cls, group: h5py.Group) -> Any:
        if 'data' in group:
            try:
                return jsonpickle.decode(group['data'][()])
            except Exception as e:
                logging.error(f"Failed to decode solver results: {e}")
        return None
    
    def to_hdf5(self, path: str, metadata: dict | None = None):
        def encode_model(model: Model, group: h5py.Group, save_instance=False):
            params_tree, static_tree = model.partition(include_fixed=True, param_objects=True)
            params = model.params()
            model_raw_grp = group.create_group('raw')
            model_raw_grp.create_dataset('params', data=jsonpickle.encode(params_tree))
            model_raw_grp.create_dataset('static', data=jsonpickle.encode(static_tree))
            
            params_grp = group.create_group('params')
            for name, initial_param in params.items():
                params_grp[name] = initial_param.to_json()
        
        with h5py.File(path, 'w') as f:
            # Metadata
            metadata_grp = f.create_group('metadata')
            fitter_metadata_grp = metadata_grp.create_group('fitter')
            fitter_metadata_grp['version'] = self.version
            fitter_metadata_grp['fit_results_cls'] = str(self.__class__.__module__ + "." + self.__class__.__qualname__)
            if self.solver_results is not None:
                fitter_metadata_grp['solver_results_cls'] = self.solver_results.__module__ + "." + self.__class__.__qualname__
            
            if not metadata is None:
                user_metadata_grp = metadata_grp.create_group('user')
                for k, v in metadata.items():
                    user_metadata_grp[k] = json.dumps(v)

            # Model fit
            if self.fit_model is not None:
                encode_model(self.fit_model, f.create_group('fit_model'))

            # Solver results
            if self.solver_results is not None:
                solver_results_grp = f.create_group('solver_results')
                self.encode_solver_results(solver_results_grp)                

            # Other input
            ## Setup
            input_grp = f.create_group('input')
            if self.initial_model is not None:
                encode_model(self.initial_model, input_grp.create_group('model'), save_instance=True)
                    
            ## Measured data
            if self.measured is not None:
                measured_grp = input_grp.create_group('measured')
                if isinstance(self.measured, skrf.Network):
                    measured_grp['name'] = self.measured.name
                    measured_grp.create_dataset('s', data=self.measured.s)
                    measured_grp.create_dataset('f', data=self.measured.f)
                    measured_grp.create_dataset('z0', data=self.measured.z0)
                else:
                    for label, ntwk in self.measured.items():
                        measured_ntwk_grp = measured_grp.create_group(label)
                        measured_ntwk_grp['name'] = ntwk.name
                        measured_ntwk_grp.create_dataset('s', data=ntwk.s)
                        measured_ntwk_grp.create_dataset('f', data=ntwk.f)
                        measured_ntwk_grp.create_dataset('z0', data=ntwk.z0)

            ## Other settings
            if self.frequency is not None:
                frequency_grp = input_grp.create_group('frequency')
                frequency_grp['f'] = self.frequency.f
                frequency_grp['unit'] = self.frequency.unit
            if self.features is not None:
                input_grp.create_dataset('features', data=json.dumps(self.features))
            if self.fit_args is not None:
                input_grp.create_dataset('fit_args', data=jsonpickle.encode(self.fit_args))
            if self.fit_kwargs is not None:
                input_grp.create_dataset('fit_kwargs', data=jsonpickle.encode(self.fit_kwargs))            
            if self.solver_args is not None:
                input_grp.create_dataset('solver_args', data=jsonpickle.encode(self.solver_args))
            if self.solver_kwargs is not None:
                input_grp.create_dataset('solver_kwargs', data=jsonpickle.encode(self.solver_kwargs))            

    @classmethod
    def from_hdf5(cls, path: str) -> "FitResults":
        def decode_model(group: h5py.Group) -> Model:
            model_raw_grp = group['raw']
            params_json = model_raw_grp['params'][()]
            params_json = params_json.decode('utf-8') if isinstance(params_json, bytes) else params_json
            static_json = model_raw_grp['static'][()]
            static_json = static_json.decode('utf-8') if isinstance(static_json, bytes) else static_json
            
            try:
                params_tree = jsonpickle.decode(params_json)
                static_tree = jsonpickle.decode(static_json)
                
                # NB the following hack actually also BREAKS some model loading... we need to investigate further
                # The following fixes some quirks when e.g. the original model contains lambdas.
                # Not sure 100% why but some fields seem to be in a "bad" state when jsonpickle cant deserialize them
                # params_tree = dataclasses.replace(params_tree)
                # static_tree = dataclasses.replace(static_tree)
                
                return eqx.combine(params_tree, static_tree)
            except:
                return None

        with h5py.File(path, 'r') as f:
            # Metadata
            if 'metadata' in f:
                metadata_grp = f['metadata']

                fitter_metadata_grp = metadata_grp['fitter']
                version = fitter_metadata_grp['version'][()]
                fit_results_cls_path = fitter_metadata_grp['fit_results_cls'][()]
                fit_results_cls_path = fit_results_cls_path.decode('utf-8') if isinstance(fit_results_cls_path, bytes) else fit_results_cls_path
                try:
                    cls = load_class_from_string(fit_results_cls_path)
                except ImportError:
                    logging.warning(f"Could not import class from path '{fit_results_cls_path}'. Using FitResults instead.")            

            # Model fit
            if version == 1:
                model = decode_model(f['model']) if 'model' in f else None
            elif version == 2:
                model = decode_model(f['fit_model']) if 'fit_model' in f else None
            
            # Solver results
            solver_results = cls.decode_solver_results(f['solver_results']) if 'solver_results' in f else None

            # Input
            input_grp = f['input']
            
            ## Initial model
            initial_model = decode_model(input_grp['model']) if 'model' in input_grp else None

            ## Measured networks
            measured = None
            if 'measured' in input_grp:
                measured_grp = input_grp['measured']
                if 'name' in measured_grp:
                    net_grp = measured_grp
                    name = net_grp['name'][()]
                    name = name.decode('utf-8') if isinstance(name, bytes) else name
                    s = net_grp['s'][()]
                    f_data = net_grp['f'][()]
                    z0 = net_grp['z0'][()]
                    measured = skrf.Network(s=s, f=f_data, z0=z0, name=name)
                else:
                    measured = {}
                    for label in measured_grp.keys():
                        net_grp = measured_grp[label]
                        name = net_grp['name'][()]
                        name = name.decode('utf-8') if isinstance(name, bytes) else name
                        s = net_grp['s'][()]
                        f_data = net_grp['f'][()]
                        z0 = net_grp['z0'][()]
                        network = skrf.Network(s=s, f=f_data, z0=z0, name=name)
                        measured[str(label)] = network

            ## Frequency and features
            frequency = None
            features = None
            if 'frequency' in input_grp:
                freq_grp = input_grp['frequency']
                f_arr = freq_grp['f'][()]
                unit = freq_grp['unit'][()]
                unit = unit.decode('utf-8') if isinstance(unit, bytes) else unit
                frequency = Frequency(f=f_arr, unit=unit)
            if 'features' in input_grp:
                features = json.loads(input_grp["features"][()])

            ## Solver args, kwargs and fit args, kwargs
            solver_args, solver_kwargs = None, None
            fit_args, fit_kwargs = None, None
            if 'solver_args' in input_grp:
                solver_args = jsonpickle.decode(input_grp['solver_args'][()])
            if 'solver_kwargs' in input_grp:
                solver_kwargs = jsonpickle.decode(input_grp['solver_kwargs'][()])
            if 'fit_args' in input_grp:
                fit_args = jsonpickle.decode(input_grp['fit_args'][()])
            if 'fit_kwargs' in input_grp:
                fit_kwargs = jsonpickle.decode(input_grp['fit_kwargs'][()])
                
            return cls(
                model=model,
                initial_model=initial_model,
                frequency=frequency,
                measured=measured,
                features=features,
                logger=None,  # Not saved/restored
                solver_results=solver_results,
                fit_args=fit_args,
                fit_kwargs=fit_kwargs,
                solver_args=solver_args,
                solver_kwargs=solver_kwargs,
                version=version,
            )              
    
def is_frequentist(solver) -> bool:
    from pmrf.fitting._frequentist import FrequentistFitter
    cls = get_fitter_class(solver)
    return issubclass(cls, FrequentistFitter)

def is_bayesian(solver) -> bool:
    from pmrf.fitting._bayesian import BayesianFitter
    cls = get_fitter_class(solver)
    return issubclass(cls, BayesianFitter)

def get_fitter_class(solver: str):
    class_name = ''.join(part[0].upper() + part[1:] for part in solver.split('-'))
    class_name = class_name + 'Fitter'
    try:
        for submodule_name, _ in iter_submodules('pmrf.fitting.fitters'):
            fitter_submodel = importlib.import_module(submodule_name)
            if hasattr(fitter_submodel, class_name):
                return getattr(fitter_submodel, class_name)
    except (ImportError, AttributeError):
        raise Exception(f'Could not find solver named {solver}')