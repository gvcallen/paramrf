from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
import pkgutil
import logging
from typing import Any

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

from pmrf._model import Model
from pmrf._frequency import Frequency
from pmrf._constants import FeatureT
from pmrf._util import LevelFilteredLogger, iter_submodules
from pmrf._model import Model
from pmrf._frequency import Frequency
from pmrf._constants import FeatureT, FeatureListT
from pmrf.fitting._features import extract_features, create_stacked_features

def Fitter(
    solver: str,
    *args,
    **kwargs
) -> 'BaseFitter':
    """Fitter factory function.
    
    This allows the creator of a fitter by simply specifying the solver type and having all arguments forwarded.
    See the relevant fitter classes for detailed documentation.

    Args:
        solver (str): The solver to use, specified as either e.g. 'ScipyMinimize' or 'scipy-minimize'.

    Returns:
        BaseFitter: The concrete fitter instance.
    """
    cls = get_fitter_class(solver)
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
    
@dataclass
class FitResults:
    model: Model | None = None
    frequency: Frequency | None = None
    measured: skrf.Network | list[skrf.Network] | None = None
    features: list[FeatureT] | None = None
    logger: logging.Logger | None = None
    solver_results: Any = None
    solver_args: tuple | None = None
    solver_kwargs: dict | None = None
    version: int = 1
    
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
        with h5py.File(path, 'w') as f:
            metadata_grp = f.create_group('metadata')
            metadata_grp['user'] = json.dumps(metadata)
            metadata_grp['version'] = self.version
            metadata_grp['cls_path'] = str(self.__class__.__module__ + "." + self.__class__.__qualname__)

            # Model
            if self.model is not None:
                params_tree, static_tree = self.model.partition(include_fixed=True, param_objects=True)
                params = self.model.params(include_fixed=True)
                
                model_grp = f.create_group('model')
                model_tree_grp = model_grp.create_group('raw')
                model_tree_grp.create_dataset('params', data=jsonpickle.encode(params_tree))
                model_tree_grp.create_dataset('static', data=jsonpickle.encode(static_tree))
                param_grp = model_grp.create_group('params')
                for name, param in params.items():
                    param_grp[name] = param.to_json()
                    
            # Measured data
            if self.measured is not None:
                measured_grp = f.create_group('measured')
                networks = self.measured if isinstance(self.measured, list) else [self.measured]
                for i, net in enumerate(networks):
                    net_grp = measured_grp.create_group(f'network_{i}')
                    net_grp['name'] = net.name
                    net_grp.create_dataset('s', data=net.s)
                    net_grp.create_dataset('f', data=net.f)
                    net_grp.create_dataset('z0', data=net.z0)

            # Fit settings
            settings_grp = f.create_group('settings')
            if self.frequency is not None:
                frequency_grp = settings_grp.create_group('frequency')
                frequency_grp['f'] = self.frequency.f
                frequency_grp['unit'] = self.frequency.unit
            if self.features is not None:
                settings_grp.create_dataset('features', data=json.dumps(self.features))

            # Solver results
            solver_grp = f.create_group('solver')
            if self.solver_results is not None:
                solver_grp['cls_path'] = self.solver_results.__module__ + "." + self.__class__.__qualname__
                solver_results_grp = solver_grp.create_group('results')
                self.encode_solver_results(solver_results_grp)
            if self.solver_args is not None:
                solver_grp.create_dataset('args', data=jsonpickle.encode(self.solver_args))
            if self.solver_kwargs is not None:
                solver_grp.create_dataset('kwargs', data=jsonpickle.encode(self.solver_kwargs))

    @classmethod
    def from_hdf5(cls, path: str) -> "FitResults":
        with h5py.File(path, 'r') as f:
            # Load metadata
            if 'metadata' in f:
                metadata_grp = f['metadata']
                version = metadata_grp['version'][()]
                cls_path = metadata_grp['cls_path'][()]
                cls_path = cls_path.decode('utf-8') if isinstance(cls_path, bytes) else cls_path
                try:
                    cls = load_class_from_string(cls_path)
                except ImportError:
                    logging.warning(f"Could not import class from path '{cls_path}'. Using FitResults instead.")            

            # Load model
            model = None
            if 'model' in f:
                model_grp = f['model']
                if 'raw' in model_grp:
                    model_raw_grp = model_grp['raw']
                    params_json = model_raw_grp['params'][()]
                    static_json = model_raw_grp['static'][()]
                    if isinstance(params_json, bytes):
                        params_json = params_json.decode('utf-8')
                    if isinstance(static_json, bytes):
                        static_json = static_json.decode('utf-8')
                    params_tree = jsonpickle.decode(params_json)
                    static_tree = jsonpickle.decode(static_json)
                    model = eqx.combine(params_tree, static_tree)

            # Load measured networks
            measured = None
            if 'measured' in f:
                measured_grp = f['measured']
                measured = []
                keys = sorted(measured_grp.keys(), key=lambda x: int(x.split('_')[1]))
                for key in keys:
                    net_grp = measured_grp[key]
                    name = net_grp['name']
                    name = name.decode('utf-8') if isinstance(name, bytes) else name
                    s = net_grp['s'][()]
                    f_data = net_grp['f'][()]
                    z0 = net_grp['z0'][()]
                    net = skrf.Network(s=s, f=f_data, z0=z0, name=name)
                    measured.append(net)
                if len(measured) == 1:
                    measured = measured[0]

            # Load frequency and features
            frequency = None
            features = None
            if 'settings' in f:
                settings_grp = f['settings']
                if 'frequency' in settings_grp:
                    freq_grp = settings_grp['frequency']
                    f_arr = freq_grp['f'][()]
                    unit = freq_grp['unit'][()]
                    frequency = Frequency(f=f_arr, unit=unit)
                if 'features' in settings_grp:
                    features = json.loads(settings_grp["features"][()])

            # Load solver results, args, kwargs
            solver_results = None
            solver_args = None
            solver_kwargs = None
            if 'solver' in f:
                solver_grp = f['solver']
                if 'results' in solver_grp:
                    solver_results = cls.decode_solver_results(solver_grp['results'])
                if 'args' in solver_grp:
                    solver_args = jsonpickle.decode(solver_grp['args'][()])
                if 'kwargs' in solver_grp:
                    solver_kwargs = jsonpickle.decode(solver_grp['kwargs'][()])
                
            return cls(
                model=model,
                frequency=frequency,
                measured=measured,
                features=features,
                logger=None,  # Not saved/restored
                solver_results=solver_results,
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
    class_name = ''.join(part.capitalize() for part in solver.split('-'))
    class_name = class_name + 'Fitter'
    try:
        for submodule_name, _ in iter_submodules('pmrf.fitting.fitters'):
            fitter_submodel = importlib.import_module(submodule_name)
            if hasattr(fitter_submodel, class_name):
                return getattr(fitter_submodel, class_name)
    except (ImportError, AttributeError):
        return None