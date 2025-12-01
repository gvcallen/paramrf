from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import Counter
import importlib
import logging
from typing import Any, Sequence
from io import BytesIO
from functools import partial

import matplotlib.pyplot as plt
import jax
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

from pmrf.models import Model, DictModel
from pmrf.frequency import Frequency
from pmrf.constants import FeatureT
from pmrf._util import LevelFilteredLogger, iter_submodules, load_class_from_string
from pmrf.frequency import Frequency, MULTIPLIER_DICT
from pmrf.constants import FeatureInputT
from pmrf import extract_features, wrap

def Fitter(
    name: str,
    *args,
    **kwargs
) -> 'BaseFitter':
    """Fitter factory function.
    
    This allows the creator of a fitter by simply specifying the fitter type and having all arguments forwarded.
    See the relevant fitter classes for detailed documentation.

    Args:
        name (str): The name of the fitter to create, specified as either e.g. 'ScipyMinimize' or 'scipy-minimize'.

    Returns:
        BaseFitter: The concrete fitter instance.
    """
    cls = get_fitter_class(name)
    return cls(*args, **kwargs)
    
@dataclass
class FitSettings:
    frequency: Frequency | None = None
    features: list[FeatureT] | None = None
    fitter_kwargs: dict | None = None
    solver_kwargs: dict | None = None    

class BaseFitter(ABC):
    """
    An abstract base class that provides the foundation for all fitting algorithms in `pmrf`.
    """
    def __init__(
        self,
        model: Model | dict[str, Model],
        measured: str | skrf.Network | dict[str, skrf.Network],
        frequency: Frequency | None = None,
        features: FeatureInputT | None = None,
    ) -> None:
        """Initializes the BaseFitter.

        Args:
            model (Model | dict[str, Model]):                           The parametric `pmrf` model to be fitted.
                                                                        A dict can optionally be passed, in which case
                                                                        the keys of the networks can be referenced during
                                                                        feature extraction by also specifying features as a dictionary.
                                                                        See the documentation for the `features` argument below.
            measured (str | skrf.Network | dict[str, skrf.Network]):    The measured network data to fit the model against.
                                                                        A dict can optionally be passed, in which case
                                                                        features can be extracted as for `model`.
            frequency (Frequency | None, optional):                     The frequency axis to perform the fit on. If `None`, the frequency
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
                                                                        Defaults to `None`, in which case real and imaginary features for all ports are used.
        """
        if isinstance(measured, str):
            measured = skrf.Network(measured)
        
        # Set the default features and ensure it is not a scalar
        features = features if features is not None else [port_feature for m, n in model.port_tuples for port_feature in (f's{m+1}{n+1}_re', f's{m+1}{n+1}_im')]
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
        
        # Initialize settings
        self.frequency = frequency or Frequency.from_skrf(measured_freq)
        self.features = features
        
        # Initialize model parameters from user and store in flat array
        if isinstance(model, dict):
            model = DictModel(model)

        self.initial_model: Model = model
        self.measured: skrf.Network | dict[str, skrf.Network] = measured
        self.measured_features = extract_features(measured, None, features)
        if rank == 0:
            self.logger = logging.getLogger("pmrf.fitting")
        else:
            self.logger = LevelFilteredLogger(null_level=logging.WARNING)
    
    def _settings(self, solver_kwargs=None, fitter_kwargs=None) -> FitSettings:
        return FitSettings(frequency=self.frequency, features=self.features, fitter_kwargs=fitter_kwargs, solver_kwargs=solver_kwargs)
    
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
        general_feature_fn = wrap(extract_features, self.initial_model, self.frequency, as_numpy=as_numpy)
        feature_fn = lambda theta: general_feature_fn(theta, self.features)
        return jax.jit(feature_fn)
    
@dataclass
class FitResults:
    measured: skrf.Network | dict[str, skrf.Network] | None = None
    initial_model: Model | None = None
    fitted_model: Model | None = None
    solver_results: Any = None
    settings: FitSettings | None = None

    def plot_s_db(self):
        """
        Plots the S-parameters (Magnitude in dB) of the Measured vs Fitted data.
        Handles both single Network and dictionary of Networks.
        """
        if self.measured is None or self.fitted_model is None:
            print("Missing measured data or fitted model.")
            return

        if self.settings is None:
            print("Missing settings (frequency data) to generate fitted networks.")
            return

        # 1. Normalize input into a list of tuples: (Name, Measured_Network, Fitted_Network)
        data_to_plot = []
        
        if isinstance(self.measured, dict):
            for key, meas_nw in self.measured.items():
                # Retrieve the specific sub-model using the key name
                try:
                    sub_model = getattr(self.fitted_model, key)
                    fit_nw = sub_model.to_skrf(self.settings.frequency)
                    data_to_plot.append((key, meas_nw, fit_nw))
                except AttributeError:
                    print(f"Warning: Could not find sub-model attribute '{key}' in fitted_model.")
        else:
            # Single network case
            fit_nw = self.fitted_model.to_skrf(self.settings.frequency)
            data_to_plot.append(("Main Model", self.measured, fit_nw))

        if not data_to_plot:
            return

        # 2. Determine Grid Dimensions
        # Rows = Number of Datasets (keys)
        # Cols = Number of S-parameters (N_ports^2)
        n_rows = len(data_to_plot)
        
        # We assume the first network is representative of port counts for layout purposes,
        # but we will handle variable ports safely in the loop.
        max_ports = max(d[1].number_of_ports for d in data_to_plot)
        n_cols = max_ports * max_ports
        
        # Create Figure
        # Adjust figsize: approx 4 inches per subplot width, 3.5 inches per row height
        fig, axes = plt.subplots(
            nrows=n_rows, 
            ncols=n_cols, 
            figsize=(4 * n_cols, 3.5 * n_rows), 
            squeeze=False, # Ensures axes is always a 2D array
            constrained_layout=True
        )

        # 3. Plotting Loop
        for row_idx, (label, meas, fit) in enumerate(data_to_plot):
            n_ports = meas.number_of_ports
            
            # Loop through all S-parameter combinations (S11, S12, S21, S22...)
            # We flatten the port grid (i, j) into the subplot row
            plot_col_idx = 0
            
            for i in range(n_ports):
                for j in range(n_ports):
                    ax = axes[row_idx, plot_col_idx]               
                    
                    # Plot Fitted (Solid line)
                    # Ensure the fitted network has the same ports or handle gracefully
                    if i < fit.number_of_ports and j < fit.number_of_ports:
                        fit.plot_s_db(m=i, n=j, ax=ax, label="Model")

                    # Plot Measured (Dashed line)
                    meas.plot_s_db(m=i, n=j, ax=ax, label="Measured", linestyle='--', color='k')
                    
                    # Visual Polish
                    s_param_label = f"S{i+1}{j+1}"
                    ax.set_title(f"{label} - {s_param_label}")
                    ax.grid(True, which="major", linestyle="-", alpha=0.5)
                    
                    # Only add legend to the first plot of the row to reduce clutter
                    if plot_col_idx == 0:
                        ax.legend(fontsize='small')
                    else:
                        # Clean up redundant legends if scikit-rf adds them automatically
                        ax.get_legend().remove() if ax.get_legend() else None

                    plot_col_idx += 1

            # Hide any unused subplots in this row (if this network has fewer ports than the max)
            for k in range(plot_col_idx, n_cols):
                axes[row_idx, k].axis('off')

        plt.show()
   
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
    
    def save_hdf(self, path: str, metadata: dict | None = None):
        version = 4
        
        with h5py.File(path, 'w') as f:
            # Metadata
            metadata_grp = f.create_group('metadata')
            internal_metadata_grp = metadata_grp.create_group('__pmrf__')
            internal_metadata_grp['fit_results_cls'] = str(self.__class__.__module__ + "." + self.__class__.__qualname__)
            internal_metadata_grp['version'] = version
            if self.solver_results is not None:
                internal_metadata_grp['solver_results_cls'] = self.solver_results.__module__ + "." + self.__class__.__qualname__
            
            if not metadata is None:
                def save_dict_to_group(d: dict, group: h5py.Group):
                    for k, v in d.items():
                        if isinstance(v, dict):
                            subgrp = group.create_group(k)
                            save_dict_to_group(v, subgrp)
                        else:
                            group[k] = json.dumps(v)        
                save_dict_to_group(metadata, metadata_grp)

            # Models
            if self.fitted_model is not None:
                self.fitted_model.write_hdf(f.create_group('fitted_model'))
            if self.initial_model is not None:
                self.initial_model.write_hdf(f.create_group('initial_model'))

            # Measured data
            if self.measured is not None:
                measured_grp = f.create_group('measured')
                if isinstance(self.measured, skrf.Network):
                    measured_grp['name'] = self.measured.name or 'ntwk'
                    measured_grp.create_dataset('s', data=self.measured.s)
                    measured_grp.create_dataset('f', data=self.measured.f)
                    measured_grp.create_dataset('z0', data=self.measured.z0)
                else:
                    for label, ntwk in self.measured.items():
                        measured_ntwk_grp = measured_grp.create_group(label)
                        measured_ntwk_grp['name'] = ntwk.name or 'ntwk'
                        measured_ntwk_grp.create_dataset('s', data=ntwk.s)
                        measured_ntwk_grp.create_dataset('f', data=ntwk.f)
                        measured_ntwk_grp.create_dataset('z0', data=ntwk.z0)
                        
            # Solver results
            if self.solver_results is not None:
                solver_results_grp = f.create_group('solver_results')
                self.encode_solver_results(solver_results_grp)                

            # Other input
            ## Setup
            input_grp = f.create_group('settings')                    

            ## Other settings
            if self.settings.frequency is not None:
                frequency_grp = input_grp.create_group('frequency')
                frequency_grp['f'] = self.settings.frequency.f
                frequency_grp['f_scaled'] = self.settings.frequency.f_scaled
                frequency_grp['unit'] = self.settings.frequency.unit
            if self.settings.features is not None:
                input_grp.create_dataset('features', data=json.dumps(self.settings.features))
            if self.settings.fitter_kwargs is not None:
                input_grp.create_dataset('fitter_kwargs', data=jsonpickle.encode(self.settings.fitter_kwargs))            
            if self.settings.solver_kwargs is not None:
                input_grp.create_dataset('solver_kwargs', data=jsonpickle.encode(self.settings.solver_kwargs))            

    @classmethod
    def load_hdf(cls, path: str) -> "FitResults":
        with h5py.File(path, 'r') as f:
            # Metadata
            if 'metadata' in f:
                metadata_grp = f['metadata']
                
                if 'fitter' in metadata_grp and 'version' in metadata_grp['fitter']:
                    internal_metadata_grp = metadata_grp['fitter']
                else:
                    internal_metadata_grp = metadata_grp['__pmrf__']
                
                if 'version' in internal_metadata_grp:
                    version = internal_metadata_grp['version'][()]
                else:
                    version = 4 # bug for some
                fit_results_cls_path = internal_metadata_grp['fit_results_cls'][()]
                fit_results_cls_path = fit_results_cls_path.decode('utf-8') if isinstance(fit_results_cls_path, bytes) else fit_results_cls_path
                try:
                    cls = load_class_from_string(fit_results_cls_path)
                except ImportError:
                    logging.warning(f"Could not import class from path '{fit_results_cls_path}'. Using FitResults instead.")            

            # Model fit
            if version == 1:
                fitted_model = Model.read_hdf(f['model']) if 'model' in f else None
            elif version == 2:
                fitted_model = Model.read_hdf(f['fit_model']) if 'fit_model' in f else None
            elif version >= 3:
                fitted_model = Model.read_hdf(f['fitted_model']) if 'fitted_model' in f else None
            
            # Solver results
            solver_results = cls.decode_solver_results(f['solver_results']) if 'solver_results' in f else None

            # Settings
            if version <= 3:
                settings_grp = f['input']
            else:
                settings_grp = f['settings']
                
            if version <= 3:            
                initial_model = Model.read_hdf(settings_grp['model']) if 'model' in settings_grp else None
            else:
                initial_model = Model.read_hdf(f['initial_model']) if 'initial_model' in f else None

            ## Measured networks
            measured = None
            measured_grp = None
            if version <= 3 and 'measured' in settings_grp:
                measured_grp = settings_grp['measured']
            elif 'measured' in f:
                measured_grp = f['measured']
            
            if measured_grp is not None:
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
            if 'frequency' in settings_grp:
                freq_grp = settings_grp['frequency']
                unit = freq_grp['unit'][()]
                unit = unit.decode('utf-8') if isinstance(unit, bytes) else unit
                if 'f_scaled' in freq_grp:
                    f_scaled_arr = freq_grp['f_scaled'][()]
                    frequency = Frequency.from_f(f=f_scaled_arr, unit=unit)
                else:
                    f_arr = freq_grp['f'][()]
                    frequency = Frequency.from_f(f_arr / MULTIPLIER_DICT[unit.lower()], unit=unit)
            if 'features' in settings_grp:
                features = json.loads(settings_grp["features"][()])

            ## Solver args, kwargs and fit args, kwargs
            solver_kwargs, fitter_kwargs = None, None
            if 'solver_kwargs' in settings_grp:
                solver_kwargs = jsonpickle.decode(settings_grp['solver_kwargs'][()])
            if 'fitter_kwargs' in settings_grp:
                fitter_kwargs = jsonpickle.decode(settings_grp['fitter_kwargs'][()])
                
            settings = FitSettings(frequency, features, fitter_kwargs, solver_kwargs)
            return cls(
                measured=measured,
                initial_model=initial_model,
                fitted_model=fitted_model,
                solver_results=solver_results,
                settings=settings,
            )
    
def is_frequentist(solver) -> bool:
    from pmrf.fitting._frequentist import FrequentistFitter
    cls = get_fitter_class(solver)
    return issubclass(cls, FrequentistFitter)

def is_bayesian(solver) -> bool:
    from pmrf.fitting._bayesian import BayesianFitter
    cls = get_fitter_class(solver)
    return issubclass(cls, BayesianFitter)

def is_inference_kind(solver, inference: str):
    if inference == 'frequentist':
        return is_frequentist(solver)
    elif inference == 'bayesian':
        return is_bayesian(solver)
    else:
        raise Exception(f"Unknown inference type '{inference}'")

def get_fitter_class(solver: str):
    solver = solver.replace('scipy', 'sciPy')
    solver = solver.replace('polychord', 'polyChord')

    class_names = [solver + 'Fitter']
    class_names.append(''.join(part[0].upper() + part[1:] for part in solver.split('-')) + 'Fitter')
    try:
        for submodule_name, _ in iter_submodules('pmrf.fitting.fitters'):
            try:
                fitter_submodel = importlib.import_module(submodule_name)
            except:
                continue
            for class_name in class_names:
                if hasattr(fitter_submodel, class_name):
                    return getattr(fitter_submodel, class_name)
    except (ImportError, AttributeError):
        raise Exception(f'Could not find solver named {solver}')