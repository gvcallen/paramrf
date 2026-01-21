import glob
from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
import logging
from typing import Any, Sequence, Callable
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import jax
import json
import skrf
import h5py
import jsonpickle
from skrf import Network

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.constants import FeatureT
from pmrf._util import LevelFilteredLogger, iter_submodules, load_class_from_string, RANK
from pmrf.frequency import Frequency, MULTIPLIER_DICT
from pmrf.constants import FeatureInputT
from pmrf import extract_features, wrap
from pmrf.network_collection import NetworkCollection

INIT_PARAMS = ['features', 'output_path', 'output_root', 'sparam_kind', 'cost_kind', 'cost_function' 'likelihood_kind', 'likelihood_params', 'feature_sigmas']
    
@dataclass
class FitSettings:
    """
    Configuration settings for the fitting process.

    Attributes
    ----------
    frequency : Frequency or None
        The frequency grid used for the fit.
    features : list of FeatureT or None
        The list of features extracted for the fit.
    fitter_kwargs : dict or None
        Keyword arguments passed to the specific fitter backend.
    solver_kwargs : dict or None
        Keyword arguments passed to the numerical solver.
    """
    frequency: Frequency | None = None
    features: list[FeatureT] | None = None
    fitter_kwargs: dict | None = None
    solver_kwargs: dict | None = None    

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

class BaseFitter(ABC):
    """
    An abstract base class that provides the foundation for all fitting algorithms in `pmrf`.
    """
    def __init__(
        self,
        model: Model,        
        *,
        features: FeatureInputT | None = None,
        output_path: str | None = None,
        output_root: str = 'fit',
        sparam_kind: str = 'all',
    ) -> None:
        """
        Initializes the BaseFitter.

        Parameters
        ----------
        model : Model
            The parametric `pmrf` Model to be fitted.                                                                            
        features : FeatureInputT or None, optional
            Defines the features to be extracted from the model and network(s).
            Defaults to `None`, in which case real and imaginary features for all ports are used.
            Can be a single feature e.g. 's11', a list of features (e.g., `['s11', 's11_mag']`),
            or a dictionary with either of the above as value. In the dictionary case,
            keys must be network names in the collection passed by `measured` during fitting, which must also
            correspond to submodels which are attributes of the model. For example,
            {'name1', ('s11'), {'name2', ('s21')} can be passed.
            Note that if a collection of networks is passed, but a feature dictionary is not,
            it is assumed that those feature(s) should be extract for each networks/submodel.
            See `extract_features(..)` more details.
        output_path : str or None
            The path for fitters to write output data to. Defaults to `None`.
        output_root : str or None
            The root name used for output files in the output path. Defaults to `None`.
        sparam_kind : str or None
            The S-parameter data kind to use for port-expansion in feature extraction. Can either be 'transmission', 'reflection' or 'all'.
            See `extract_features` for more details.
        """
        # Populate parameters
        self.model: Model = model
        self.features: FeatureInputT | None = features
        self.output_path = output_path
        self.output_root = output_root
        self.sparam_kind = sparam_kind
        
        if RANK == 0:
            self.logger = logging.getLogger("pmrf.fitting")
        else:
            self.logger = LevelFilteredLogger(null_level=logging.WARNING)
            
    def fit(
        self,
        measured: str | Network | NetworkCollection,
        **kwargs         
    ) -> 'FitResults':
        """
        Fits the model to measured data.

        This method fits the full model using the original features specified.

        Arguments are forwarded to `self.run(...)`.

        Parameters
        ----------
        measured : skrf.Network or prf.NetworkCollection
            The measured network data to fit the model against.
            Can be a scikit-rf `Network` or a paramrf `NetworkCollection`.
            For the latter case the network names should be referenced during
            feature extraction by specifying features as a dictionary.
            If networks do not have the same frequency, a common frequency is used.
        **kwargs
            Additional arguments forwarded to `self.run`.

        Returns
        -------
        FitResults
            The fit results.
        """
        if isinstance(measured, str):
            measured = skrf.Network(measured)
        
        ctx = self.create_context(measured)
        results = self.run(ctx, **kwargs)

        return results
    
    def fit_submodels(
        self,
        measured: NetworkCollection,
        **kwargs         
    ) -> 'FitResults':
        """
        Fits the submodels.
         
        This method fits the model to the measured data by fitting its submodels in a sequential manner.

        Arguments are forwarded `self.run(...)`.

        Parameters
        ----------
        measured : prf.NetworkCollection
            The measured network data to fit the model against.
            Must be a ParamRF `NetworkCollection`. Network names should be referenced during
            feature extraction by specifying features as a dictionary.
            If networks do not have the same frequency, a common frequency is used.
        **kwargs
            Additional arguments forwarded to `self.run`.

        Returns
        -------
        FitResults
            The fit results. `solver_results` contains a dictionary of the individual submodel results.
        """
        all_results: dict[str, FitResults] = {}
        
        # Fit the components sequentially
        for ntwk in measured:
            name = ntwk.name
            
            self.logger.info(f'Fitting {name}...')
            
            model = self.model.with_free_submodels([name], fix_others=True)
            comp_measured = measured.filter(lambda ntwk: ntwk.name == name)
            output_path = f'{self.output_path}/submodels/{name}' if self.output_path is not None else None
            
            ctx = self.create_context(comp_measured, model=model, output_path=output_path)
            all_results[name] = self.run(ctx, **kwargs)

        fitted_model = self.model.with_models([result.fitted_model for result in all_results.values()])
        fit_results = FitResults(
            initial_model=self.model,
            fitted_model=fitted_model,
            solver_results=all_results,

        )
        metadata = fitted_model.metadata
        metadata['fit_results'] = fit_results
        fitted_model = fitted_model.with_fields(metadata=metadata)
        
        if RANK == 0 and self.output_path is not None:
            if kwargs.get('save_model', True):
                name = fitted_model.name or 'model'
                fitted_model.save(f'{self.output_path}/fitted_{name}.prf')
            if kwargs.get('save_results', True):
                fit_results.save_hdf(f'{self.output_path}/results.hdf5')

        return fit_results

    def create_context(self, measured, *, model=None, features=None, output_path=None, output_root=None, sparam_kind=None) -> FitContext:
        """
        Creates a FitContext from the provided measurement and optional overrides.

        Parameters
        ----------
        measured : skrf.Network or NetworkCollection
            The measured data.
        model : Model, optional
            Model override. Defaults to self.model.
        features : FeatureInputT, optional
            Features override. Defaults to self.features.
        output_path : str, optional
            Output path override. Defaults to self.output_path.
        output_root : str, optional
            Output root override. Defaults to self.output_root.
        sparam_kind : str, optional
            S-parameter kind override. Defaults to self.sparam_kind.

        Returns
        -------
        FitContext
            The initialized context object.
        """
        model = model or self.model
        features = features or self.features
        sparam_kind = sparam_kind or self.sparam_kind
        output_path = output_path or self.output_path
        output_root = output_root or self.output_root
        
        # Make sure measured is loaded, and that all frequencies are the same
        if isinstance(measured, str):
            measured = skrf.Network(measured)
        measured = measured.copy()
        if isinstance(measured, NetworkCollection):
            measured.interpolate_self()
            frequency = measured.common_frequency()
        else:
            frequency = measured.frequency
        frequency = Frequency.from_skrf(frequency)

        # Set the default features and ensure it is not a scalar
        features = features if features is not None else [port_feature for m, n in model.port_tuples for port_feature in (f's{m+1}{n+1}_re', f's{m+1}{n+1}_im')]
        if not isinstance(features, Sequence) and not isinstance(features, dict):
            features = [features]
        if isinstance(measured, NetworkCollection) and not isinstance(features, dict):
            features = {ntwk.name: features for ntwk in measured}

        measured_features = extract_features(measured, None, features, sparam_kind=sparam_kind)
        
        return FitContext(
            measured=measured,
            model=model,
            frequency=frequency,
            features=features,
            measured_features=measured_features,
            logger=self.logger,
            output_path=output_path,
            output_root=output_root
        )
    
    def run(
        self,
        context: FitContext,
        *,
        load_previous: bool = True, 
        new_uniform_frac: float | None = 0.01,
        save_model: bool = True,
        save_results: bool = True,
        plot_s_db: bool = True,
        callback: Callable[['FitResults'], None] | None = None,
        **kwargs
    ) -> 'FitResults':
        """
        Runs the fitting algorithm for the specific context.

        This is a low-level method and should seldom be used directly.

        This method runs the fitting algorithm implemented by the underlying sub-class.
        It contains several convenience parameters, allowing for e.g. automatic saving
        and plotting of results.

        Additional arguments are forwarded to the underlying fitter.

        Parameters
        ----------
        context: FitContext
            The fitting context.
        load_previous : bool, default=True
            Whether or not to try and load previous results from the output path.
        new_uniform_frac : float or None, optional, default=0.01
            The fraction to update model distribution bounds uniformly around the fitted model values.
        save_model : bool, default=True
            Saves the model to the output path (if provided).
        save_results : bool, default=True
            Saves the results to hdf format in the output path (if provided).
        plot_s_db : bool, default=True
            Plots the S-parameters in db and save the results in the output path (if provided]).
        callback : Callable[[FitResults], None] or None, optional
            A callback to run after fitting but before saving and plotting.
        **kwargs
            Additional arguments forwarded to the underlying fitter.

        Returns
        -------
        FitResults
            The fitted results object.
        """
        # Try load from previous results
        if load_previous and context.output_path is not None:
            try:
                filename = glob.glob(f"{context.output_path}/*.hdf5")[0]
                results = FitResults.load_hdf(filename)
                logging.info(f"Loaded previous results.")
                return results
            except:
                pass

        # Output fit parameters and features
        self.logger.info(f"Fitting for {context.model.num_flat_params} parameters")
        self.logger.info(f"Parameter names: {context.model.flat_param_names()}")
        self.logger.info(f'Features: {context.features}')
        
        results = self._run(context, **kwargs)
        results.measured = context.measured
        results.initial_model = context.model
        results.settings = context.settings(solver_kwargs=kwargs)

        if new_uniform_frac is not None:
            results.fitted_model = results.fitted_model.with_uniform_distributions(new_uniform_frac)

        if callback:
            callback(results)

        save_output = context.output_path is not None and (save_model or save_results or plot_s_db) and RANK == 0
        if save_output:
            Path(context.output_path).mkdir(parents=True, exist_ok=True)                
            if save_model:
                fitted_model = results.fitted_model
                model_name = fitted_model.name or 'model'
                fitted_model.save(f'{context.output_path}/fitted_{model_name}.prf')

            if save_results:
                results.save_hdf(f'{context.output_path}/results.hdf5')
        
            if plot_s_db:
                results.plot_s_db()
                plt.savefig(f'{context.output_path}/s_db.png', dpi=400)
                plt.close()

        model_metadata = results.fitted_model.metadata
        model_metadata['fit_results'] = results
        results.fitted_model = results.fitted_model.with_fields(metadata=model_metadata)
        
        return results
    
    @abstractmethod
    def _run(self, context: FitContext, **kwargs) -> 'FitResults':
        """
        Executes the fitting algorithm.

        This method must be implemented by all concrete subclasses. It is the
        main entry point to start the optimization or sampling process.

        Parameters
        ----------
        context : FitContext
            The context containing data and model.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        FitResults
            An object containing the results of the fit.
        """        
        raise NotImplementedError    
    
@dataclass
class FitResults:
    """
    Container for the results of a model fitting process.
    
    This class holds the state of the model before and after optimization,
    the original data, and the raw output from the solver.

    Attributes
    ----------
    measured : skrf.Network, NetworkCollection, or None
        The original measured data (target) against which the model was fit.
    initial_model : Model or None
        The model with the initial parameters.
    fitted_model : Model or None
        The model with the fitted parameters.
    solver_results : Any
        The raw result object returned by the optimization backend.
    settings : FitSettings or None
        The configuration and hyperparameters used to execute the fit.
    """    
    measured: skrf.Network | NetworkCollection | None = None
    initial_model: Model | None = None
    fitted_model: Model | None = None
    solver_results: Any = None
    settings: FitSettings | None = None 

    def plot_s_db(self, use_initial_model=False):
        """
        Plots the S-parameters (Magnitude in dB) of the Measured vs Fitted data.
        Handles both single Network and a `NetworkCollection`.

        Parameters
        ----------
        use_initial_model : bool, optional, default=False
            Whether or not to use the initial model.
        """
        model = self.initial_model if use_initial_model else self.fitted_model
        
        if self.measured is None or model is None:
            print("Missing measured data or fitted model.")
            return

        if self.settings is None:
            print("Missing settings (frequency data) to generate fitted networks.")
            return

        # 1. Normalize input into a list of tuples: (Name, Measured_Network, Fitted_Network)
        data_to_plot = []
        
        if isinstance(self.measured, NetworkCollection):
            for meas_nw in self.measured:
                # Retrieve the specific sub-model using the key name
                try:
                    sub_model = getattr(model, meas_nw.name)
                    fit_nw = sub_model.to_skrf(self.settings.frequency)
                    data_to_plot.append((meas_nw.name, meas_nw, fit_nw))
                except AttributeError:
                    print(f"Warning: Could not find sub-model attribute '{key}' in fitted_model.")
        else:
            # Single network case
            fit_nw = model.to_skrf(self.settings.frequency)
            data_to_plot.append(("Main Model", self.measured, fit_nw))

        if not data_to_plot:
            return

        # 2. Determine Grid Dimensions
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
            # constrained_layout=True
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

        fig.tight_layout()

    def encode_solver_results(self, group: h5py.Group):
        """
        Encode solver results into an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group to write to.
        """
        data = None
        if self.solver_results is not None:
            try:
                data = jsonpickle.encode(self.solver_results)
            except Exception as e:
                logging.error(f"Failed to encode solver results: {e}")
        group['data'] = data
    
    @classmethod
    def decode_solver_results(cls, group: h5py.Group) -> Any:
        """
        Decode solver results from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group to read from.

        Returns
        -------
        Any
            The decoded solver results object or None.
        """
        if 'data' in group:
            try:
                return jsonpickle.decode(group['data'][()])
            except Exception as e:
                logging.error(f"Failed to decode solver results: {e}")
        return None
    
    def save_hdf(self, path: str, metadata: dict | None = None):
        """
        Save the fit results to an HDF5 file.

        Parameters
        ----------
        path : str
            The file path to save to.
        metadata : dict, optional
            Additional metadata to save.
        """
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
            # TODO save network params
            def write_network(group: h5py.Group, ntwk: skrf.Network):
                group['name'] = ntwk.name or 'network'
                group.create_dataset('s', data=ntwk.s)
                group.create_dataset('f', data=ntwk.f)
                group.create_dataset('z0', data=ntwk.z0)
                if ntwk.params is not None:
                    measured_params_grp = group.create_group('params')
                    for key, value in ntwk.params.items():
                        measured_params_grp[key] = value

            if self.measured is not None:
                measured_grp = f.create_group('measured')
                if isinstance(self.measured, skrf.Network):
                    write_network(measured_grp, self.measured)
                else:
                    for ntwk in self.measured:
                        ntwk_grp = measured_grp.create_group(ntwk.name)
                        write_network(ntwk_grp, ntwk)
                        
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
        """
        Load fit results from an HDF5 file.

        Parameters
        ----------
        path : str
            The file path to load from.

        Returns
        -------
        FitResults
            The loaded results object.
        """
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

            def group_to_dict(group: h5py.Group):
                result = {}
                for key, item in group.items():
                    if isinstance(item, h5py.Group):
                        result[key] = group_to_dict(item)  # recurse
                    else:  # Dataset
                        result[key] = item[()]  # read dataset into memory
                return result                
            
            def read_network(group: h5py.Group):
                name = group['name'][()]
                name = name.decode('utf-8') if isinstance(name, bytes) else name
                s = group['s'][()]
                f_data = group['f'][()]
                z0 = group['z0'][()]
                if 'params' in group:
                    params = group_to_dict(group['params'])
                else:
                    params = None
                return skrf.Network(s=s, f=f_data, z0=z0, name=name, params=params)
            
            if measured_grp is not None:
                if 's' in measured_grp and 'f' in measured_grp and 'z0' in measured_grp:
                    measured = read_network(measured_grp)
                else:
                    params = group_to_dict(measured_grp['params']) if 'params' in measured_grp else None
                    measured = NetworkCollection(params=params)
                    for label in measured_grp.keys():
                        network = read_network(measured_grp[label])
                        measured.add(network)

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

def Fitter(
    model: Model,
    *,
    inference: str | None = None,
    backend: str | None = None,
    **kwargs
) -> 'BaseFitter':
    """
    Fitter factory function.
    
    This allows the creator of a fitter by simply specifying the inference type or fitter backend and having all arguments forwarded.
    See the relevant fitter classes for detailed documentation.

    Parameters
    ----------
    model : Model
        The parametric `pmrf` Model to be fitted.
        See the documentation for `BaseFitter`.
    inference : str, optional
        High-level inference mode. Can be either 'frequentist' or 'bayesian'.
        If provided and ``backend`` is ``None``, a suitable default backend
        is selected automatically.
    backend : str, optional
        Explicit fitter backend name. If provided, this takes precedence over
        ``inference`` and must be compatible with it.
    **kwargs
        Additional arguments forwarded to the fitter constructor.

    Returns
    -------
    BaseFitter
        The concrete fitter instance.
    """
    if inference is None and backend is None:
        inference = 'frequentist'
    if inference not in [None, 'frequentist', 'bayesian']:
        raise Exception('Unknown inference type')
    if backend is None:
        backend = 'scipy-minimize' if inference == 'frequentist' else 'polychord'
    
    if not is_inference_kind(backend, inference):
        raise Exception('Inference type incompatible with backend')

    cls = get_fitter_class(backend)
    return cls(model, **kwargs)

def is_frequentist(solver) -> bool:
    """
    Check if a solver is a Frequentist fitter.

    Parameters
    ----------
    solver : str
        The name of the solver.

    Returns
    -------
    bool
        True if the solver corresponds to a FrequentistFitter subclass.
    """
    from fitting.frequentist import FrequentistFitter
    cls = get_fitter_class(solver)
    return issubclass(cls, FrequentistFitter)

def is_bayesian(solver) -> bool:
    """
    Check if a solver is a Bayesian fitter.

    Parameters
    ----------
    solver : str
        The name of the solver.

    Returns
    -------
    bool
        True if the solver corresponds to a BayesianFitter subclass.
    """
    from fitting.bayesian import BayesianFitter
    cls = get_fitter_class(solver)
    return issubclass(cls, BayesianFitter)

def is_inference_kind(solver, inference: str):
    """
    Check if a solver matches a specific inference kind.

    Parameters
    ----------
    solver : str
        The name of the solver.
    inference : str
        The inference kind ('frequentist' or 'bayesian').

    Returns
    -------
    bool
        True if the solver matches the inference kind.

    Raises
    ------
    Exception
        If the inference kind is unknown.
    """
    if inference == 'frequentist':
        return is_frequentist(solver)
    elif inference == 'bayesian':
        return is_bayesian(solver)
    else:
        raise Exception(f"Unknown inference type '{inference}'")

def get_fitter_class(solver: str):
    """
    Retrieve the Fitter class corresponding to a solver name.

    Parameters
    ----------
    solver : str
        The name of the solver (e.g., 'scipy-minimize').

    Returns
    -------
    class
        The fitter class found in the backends.

    Raises
    ------
    Exception
        If the solver class cannot be found or imported.
    """
    solver = solver.replace('scipy', 'sciPy')
    solver = solver.replace('polychord', 'polyChord')

    class_names = [solver + 'Fitter']
    class_names.append(''.join(part[0].upper() + part[1:] for part in solver.split('-')) + 'Fitter')
    try:
        for submodule_name, _ in iter_submodules('pmrf.fitting._backends'):
            fitter_submodel = importlib.import_module(submodule_name)
            for class_name in class_names:
                if hasattr(fitter_submodel, class_name):
                    return getattr(fitter_submodel, class_name)
    except (ImportError, AttributeError) as e:
        raise Exception(f'Could not find solver named {solver} with error: {e}')