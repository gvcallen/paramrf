import glob
from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
import logging
from typing import Any, Sequence, Callable
import os
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
        output_root: str | None = None,
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
            The root name to prepend (with an underscore) to output files in the output path. Defaults to `None`.
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

        Arguments are forwarded to ``self.run_context(...)``.

        Parameters
        ----------
        measured : skrf.Network or prf.NetworkCollection
            The measured network data to fit the model against.
            Can be a scikit-rf `Network` or a paramrf `NetworkCollection`.
            For the latter case the network names should be referenced during
            feature extraction by specifying features as a dictionary.
            If networks do not have the same frequency, a common frequency is used.
        **kwargs
            Additional arguments forwarded to the underlying algorithm via ``self.run_context``.

        Returns
        -------
        FitResults
            The fit results object.
        """
        if isinstance(measured, str):
            measured = skrf.Network(measured)
        
        ctx = self._create_context(measured)
        results = self._run_context(ctx, **kwargs)

        return results
    
    def fit_submodels(
        self,
        measured: NetworkCollection,
        **kwargs         
    ) -> 'FitResults':
        """
        Fits the submodels.
         
        This method fits the model to the measured data by fitting its submodels in a sequential manner.

        Arguments are forwarded to ``self.run_context(...)``.

        Parameters
        ----------
        measured : prf.NetworkCollection
            The measured network data to fit the model against.
            Must be a ParamRF `NetworkCollection`. Network names should be referenced during
            feature extraction by specifying features as a dictionary.
            If networks do not have the same frequency, a common frequency is used.
        **kwargs
            Additional arguments forwarded to the underlying algorithm via ``self.run_context``.

        Returns
        -------
        FitResults
            The fit results object. `solver_results` contains a dictionary of the individual submodel results.
        """
        all_results: dict[str, FitResults] = {}
        
        # Fit the components sequentially
        ctx_kwargs = kwargs.copy()
        ctx_kwargs['save_results'] = False
        ctx_kwargs['save_model'] = False
        ctx_kwargs.setdefault('figure_subfolder', '../../figures')

        for ntwk in measured:
            name = ntwk.name
            
            self.logger.info(f'Fitting {name}...')
            
            model = self.model.with_free_submodels([name], fix_others=True)
            comp_measured = measured.filter(lambda ntwk: ntwk.name == name)
            output_path = f'{self.output_path}/submodels/{ntwk.name}'
            
            ctx = self._create_context(comp_measured, model=model, output_path=output_path, output_root=name)
            all_results[name] = self._run_context(ctx, **ctx_kwargs)

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
            self.logger.info(f'Saving combined model...')
            if kwargs.get('save_model', True):
                name = fitted_model.name or 'model'
                fitted_model.save(f'{self.output_path}/fitted_{name}.prf')
            self.logger.info(f'Saving combined results...')
            if kwargs.get('save_results', True):
                fit_results.save_hdf(f'{self.output_path}/results.hdf5')

        return fit_results

    def _create_context(self, measured, *, model=None, features=None, output_path=None, output_root=None, sparam_kind=None) -> FitContext:
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
            output_root=output_root,
            sparam_kind=sparam_kind,
        )
    
    def _run_context(
        self,
        context: FitContext,
        *,
        load_previous: bool = True, 
        new_uniform_frac: float | None = 0.01,
        save_model: bool = True,
        save_results: bool = True,
        plot_s_db: bool = True,
        figure_subfolder: str | None = None,
        callback: Callable[['FitResults'], None] | None = None,
        **kwargs
    ) -> 'FitResults':
        """
        Executes the fitting context.

        This is a low-level method and should seldom be used directly.

        This method runs the fitting algorithm implemented by the underlying sub-class.
        It contains several convenience parameters, allowing for e.g. automatic saving
        and plotting of results.

        Additional arguments are forwarded to the underlying algorithm via ``self.run_algorithm``.

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
            Additional arguments forwarded to the underlying fitter via ``self.run_algorithm``.

        Returns
        -------
        FitResults
            The fit results object.
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
        
        results = self._run_algorithm(context, **kwargs)
        results.measured = context.measured
        results.initial_model = context.model
        results.settings = context.settings(solver_kwargs=kwargs)

        if new_uniform_frac is not None:
            results.fitted_model = results.fitted_model.with_uniform_distributions(new_uniform_frac)

        if callback:
            callback(results)

        output_path = context.output_path
        save_output = context.output_path is not None and (save_model or save_results or plot_s_db) and RANK == 0
        if save_output:
            output_prefix = f'{output_path}/{context.output_root}_' if context.output_root is not None else f'{output_path}/'
            fitted_model = results.fitted_model
            name = results.measured[0].name or fitted_model.name or 'model'
            if save_model:
                Path(output_path).resolve().mkdir(parents=True, exist_ok=True)
                self.logger.info(f'Saving {name} model...')
                fitted_model.save(Path(f'{output_prefix}fitted_{name}.prf').resolve())

            if save_results:
                Path(output_path).resolve().mkdir(parents=True, exist_ok=True)
                self.logger.info(f'Saving {name} results...')
                results.save_hdf(Path(f'{output_prefix}results.hdf5').resolve())
        
            if plot_s_db:
                self.logger.info(f'Plotting {name} S-parameters in db...')

                figure_path = f'{output_path}/{figure_subfolder}' if figure_subfolder is not None else output_path
                figure_prefix = f'{figure_path}/{context.output_root}_' if context.output_root is not None else f'{figure_path}/'
                Path(figure_path).resolve().mkdir(parents=True, exist_ok=True)

                results.plot_s_db()
                plt.savefig(Path(f'{figure_prefix}s_db.png').resolve(), dpi=400)
                plt.close()

        model_metadata = results.fitted_model.metadata
        model_metadata['fit_results'] = results
        results.fitted_model = results.fitted_model.with_fields(metadata=model_metadata)
        
        return results
    
    @abstractmethod
    def _run_algorithm(self, context: FitContext, **kwargs) -> 'FitResults':
        """
        Executes the fitting algorithm.

        This is a low-level method and should seldom be used directly.

        This method must be implemented by all concrete subclasses. It is the
        main entry point to start the optimization or sampling process.

        Parameters
        ----------
        context : FitContext
            The fitting context.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        FitResults
            The fit results object.
        """        
        raise NotImplementedError    
    
@dataclass
class FitResults:
    """
    Container for the results of a model fitting process.
    
    Attributes
    ----------
    measured : skrf.Network, NetworkCollection, or None
        The original measured data (target).
    initial_model : Model or None
        The model with the initial parameters.
    fitted_model : Model or None
        The model with the fitted parameters.
    solver_results : Any
        The raw result object returned by the optimization backend.
    settings : FitSettings or None
        The configuration used to execute the fit.
    """    
    measured: skrf.Network | NetworkCollection | None = None
    initial_model: Model | None = None
    fitted_model: Model | None = None
    solver_results: Model = None
    settings: FitSettings | None = None

    # --------------------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------------------
    def plot_s_db(self, use_initial_model=False):
        """
        Plots the S-parameters (Magnitude in dB) of the Measured vs Fitted data.
        """
        import matplotlib.pyplot as plt

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
                try:
                    sub_model = getattr(model, meas_nw.name)
                    fit_nw = sub_model.to_skrf(self.settings.frequency)
                    data_to_plot.append((meas_nw.name, meas_nw, fit_nw))
                except AttributeError:
                    print(f"Warning: Could not find sub-model attribute '{meas_nw.name}' in fitted_model.")
        else:
            fit_nw = model.to_skrf(self.settings.frequency)
            data_to_plot.append(("Main Model", self.measured, fit_nw))

        if not data_to_plot:
            return

        # 2. Determine Grid Dimensions
        n_rows = len(data_to_plot)
        max_ports = max(d[1].number_of_ports for d in data_to_plot)
        n_cols = max_ports * max_ports
        
        fig, axes = plt.subplots(
            nrows=n_rows, 
            ncols=n_cols, 
            figsize=(4 * n_cols, 3.5 * n_rows), 
            squeeze=False
        )

        # 3. Plotting Loop
        for row_idx, (label, meas, fit) in enumerate(data_to_plot):
            n_ports = meas.number_of_ports
            plot_col_idx = 0
            
            for i in range(n_ports):
                for j in range(n_ports):
                    ax = axes[row_idx, plot_col_idx]              
                    
                    if i < fit.number_of_ports and j < fit.number_of_ports:
                        fit.plot_s_db(m=i, n=j, ax=ax, label="Model")

                    meas.plot_s_db(m=i, n=j, ax=ax, label="Measured", linestyle='--', color='k')
                    
                    s_param_label = f"S{i+1}{j+1}"
                    ax.set_title(f"{label} - {s_param_label}")
                    ax.grid(True, which="major", linestyle="-", alpha=0.5)
                    
                    if plot_col_idx == 0:
                        ax.legend(fontsize='small')
                    else:
                        if ax.get_legend(): ax.get_legend().remove()

                    plot_col_idx += 1

            for k in range(plot_col_idx, n_cols):
                axes[row_idx, k].axis('off')

        fig.tight_layout()

    # --------------------------------------------------------------------------
    # Public HDF5 IO Interface
    # --------------------------------------------------------------------------

    def save_hdf(self, path: str, metadata: dict | None = None):
        """
        Save the fit results to an HDF5 file.
        """
        with h5py.File(path, 'w') as f:
            self._write_to_group(f, metadata)

    @classmethod
    def load_hdf(cls, path: str) -> "FitResults":
        """
        Load fit results from an HDF5 file.
        """
        with h5py.File(path, 'r') as f:
            return cls._read_from_group(f)

    # --------------------------------------------------------------------------
    # Polymorphic Solver Results IO
    # --------------------------------------------------------------------------

    def encode_solver_results(self, group: h5py.Group):
        """
        Encode solver results into an HDF5 group.
        
        Base implementation uses recursion to handle FitResults, dicts, and lists.
        Subclasses can override this for specific formats (e.g. Scipy results).
        """
        if self.solver_results is None:
            return

        # 1. Check for recursive structures (Dict, List, or nested FitResults)
        if isinstance(self.solver_results, (dict, list, tuple, set, FitResults)):
            self._encode_recursive(self.solver_results, group)
        
        # 2. Fallback for generic objects
        else:
            try:
                data = jsonpickle.encode(self.solver_results)
                dt = h5py.special_dtype(vlen=str)
                # Store as 1-element array to avoid scalar dataset issues in some viewers
                group.create_dataset('data', data=np.array([data], dtype=dt))
            except Exception as e:
                logging.error(f"Failed to encode solver results: {e}")

    @classmethod
    def decode_solver_results(cls, group: h5py.Group) -> Any:
        """
        Decode solver results from an HDF5 group.
        """
        # 1. Recursive Structure detected
        if '__type__' in group.attrs:
            return cls._decode_recursive(group)
        
        # 2. Legacy/Generic object detected
        elif 'data' in group:
            try:
                data = group['data'][()]
                if isinstance(data, np.ndarray): data = data[0]
                if isinstance(data, bytes): data = data.decode('utf-8')
                return jsonpickle.decode(data)
            except Exception as e:
                logging.error(f"Failed to decode solver results: {e}")
                return None
        
        return None

    # --------------------------------------------------------------------------
    # Recursive Encoding Helpers (Protected)
    # --------------------------------------------------------------------------

    def _encode_recursive(self, obj: Any, group: h5py.Group):
        # Case A: Nested FitResults
        if isinstance(obj, FitResults):
            group.attrs['__type__'] = 'FitResults'
            obj._write_to_group(group)
            return

        # Case B: Dictionary
        if isinstance(obj, dict):
            group.attrs['__type__'] = 'dict'
            for k, v in obj.items():
                sub_grp = group.create_group(str(k))
                self._encode_recursive(v, sub_grp)
            return

        # Case C: Iterables
        if isinstance(obj, (list, tuple, set)):
            group.attrs['__type__'] = type(obj).__name__ 
            for i, item in enumerate(obj):
                sub_grp = group.create_group(str(i))
                self._encode_recursive(item, sub_grp)
            return

        # Case D: Leaf Node
        group.attrs['__type__'] = 'leaf'
        try:
            data = jsonpickle.encode(obj)
            dt = h5py.special_dtype(vlen=str)
            group.create_dataset('data', data=np.array([data], dtype=dt))
        except Exception as e:
            logging.error(f"Failed to encode leaf: {e}")

    @classmethod
    def _decode_recursive(cls, group: h5py.Group) -> Any:
        type_tag = group.attrs.get('__type__')

        if type_tag == 'FitResults':
            # This handles polymorphism for nested results (e.g. FitResults vs SciPyResults)
            return FitResults._read_from_group(group)

        elif type_tag in ('list', 'tuple', 'set'):
            keys = sorted(group.keys(), key=lambda x: int(x) if x.isdigit() else x)
            items = [cls._decode_recursive(group[k]) for k in keys]
            if type_tag == 'tuple': return tuple(items)
            if type_tag == 'set': return set(items)
            return items

        elif type_tag == 'dict':
            res = {}
            for k in group.keys():
                res[k] = cls._decode_recursive(group[k])
            return res

        else: # Leaf or unknown
            if 'data' in group:
                data = group['data'][()]
                if isinstance(data, np.ndarray): data = data[0]
                if isinstance(data, bytes): data = data.decode('utf-8')
                return jsonpickle.decode(data)
            return None

    # --------------------------------------------------------------------------
    # Internal IO Drivers
    # --------------------------------------------------------------------------

    def _write_to_group(self, group: h5py.Group, metadata: dict | None = None):
        """Internal driver to save full object state."""
        # 1. Metadata
        metadata_grp = group.create_group('metadata')
        internal_grp = metadata_grp.create_group('__pmrf__')
        internal_grp['fit_results_cls'] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        internal_grp['version'] = 4
        
        if metadata:
            self._save_dict_to_group(metadata, metadata_grp)

        # 2. Models
        if self.initial_model: self.initial_model.write_hdf(group.create_group('initial_model'))
        if self.fitted_model: self.fitted_model.write_hdf(group.create_group('fitted_model'))

        # 3. Measured
        if self.measured:
            meas_grp = group.create_group('measured')
            if isinstance(self.measured, skrf.Network):
                self._write_network(meas_grp, self.measured)
            else: # NetworkCollection
                for ntwk in self.measured:
                    self._write_network(meas_grp.create_group(ntwk.name), ntwk)

        # 4. Settings
        if self.settings:
            self._write_settings(group.create_group('settings'))

        # 5. Solver Results
        if self.solver_results is not None:
            self.encode_solver_results(group.create_group('solver_results'))

    @classmethod
    def _read_from_group(cls, group: h5py.Group) -> "FitResults":
        """Internal driver to load full object state."""
        
        # 1. Determine Class Type
        target_cls = cls
        if 'metadata' in group:
            meta = group['metadata'].get('__pmrf__', {}) or group['metadata'].get('fitter', {})
            if 'fit_results_cls' in meta:
                cls_path = meta['fit_results_cls'][()]
                cls_path = cls_path.decode('utf-8') if isinstance(cls_path, bytes) else cls_path
                target_cls = load_class_from_string(cls_path)

        # 2. Load Models
        fitted_model = None
        if 'fitted_model' in group:
            fitted_model = Model.read_hdf(group['fitted_model'])
        
        initial_model = None
        if 'initial_model' in group:
            initial_model = Model.read_hdf(group['initial_model'])

        # 3. Load Measured
        measured = None
        if 'measured' in group:
            measured_grp = group['measured']
            if 's' in measured_grp and 'f' in measured_grp:
                measured = cls._read_network(measured_grp)
            else:
                params = cls._group_to_dict(measured_grp['params']) if 'params' in measured_grp else None
                measured = NetworkCollection(params=params)
                for label in measured_grp.keys():
                    if label == 'params': continue
                    measured.add(cls._read_network(measured_grp[label]))

        # 4. Load Settings
        settings = None
        if 'settings' in group:
            settings = cls._read_settings(group['settings'])

        # 5. Load Solver Results
        solver_results = None
        if 'solver_results' in group:
            solver_results = target_cls.decode_solver_results(group['solver_results'])

        return target_cls(
            measured=measured,
            initial_model=initial_model,
            fitted_model=fitted_model,
            solver_results=solver_results,
            settings=settings
        )

    # --------------------------------------------------------------------------
    # Static Helper Methods
    # --------------------------------------------------------------------------

    @staticmethod
    def _save_dict_to_group(d: dict, group: h5py.Group):
        for k, v in d.items():
            if isinstance(v, dict):
                subgrp = group.create_group(k)
                FitResults._save_dict_to_group(v, subgrp)
            else:
                group[k] = json.dumps(v)

    @staticmethod
    def _group_to_dict(group: h5py.Group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                result[key] = FitResults._group_to_dict(item)
            else:
                result[key] = item[()]
        return result

    @staticmethod
    def _write_network(group: h5py.Group, ntwk: skrf.Network):
        group['name'] = ntwk.name or 'network'
        group.create_dataset('s', data=ntwk.s)
        group.create_dataset('f', data=ntwk.f)
        group.create_dataset('z0', data=ntwk.z0)
        if ntwk.params is not None:
            params_grp = group.create_group('params')
            for key, value in ntwk.params.items():
                params_grp[key] = value

    @staticmethod
    def _read_network(group: h5py.Group) -> skrf.Network:
        name = group['name'][()]
        name = name.decode('utf-8') if isinstance(name, bytes) else name
        s = group['s'][()]
        f_data = group['f'][()]
        z0 = group['z0'][()]
        params = None
        if 'params' in group:
            params = FitResults._group_to_dict(group['params'])
        return skrf.Network(s=s, f=f_data, z0=z0, name=name, params=params)

    def _write_settings(self, group: h5py.Group):
        if self.settings.frequency is not None:
            frequency_grp = group.create_group('frequency')
            frequency_grp['f'] = self.settings.frequency.f
            frequency_grp['f_scaled'] = self.settings.frequency.f_scaled
            frequency_grp['unit'] = self.settings.frequency.unit
        
        if self.settings.features is not None:
            group.create_dataset('features', data=json.dumps(self.settings.features))
        
        if self.settings.fitter_kwargs is not None:
            group.create_dataset('fitter_kwargs', data=jsonpickle.encode(self.settings.fitter_kwargs))            
        
        if self.settings.solver_kwargs is not None:
            group.create_dataset('solver_kwargs', data=jsonpickle.encode(self.settings.solver_kwargs))

    @staticmethod
    def _read_settings(group: h5py.Group):
        # Assumes FitSettings is available in scope
        frequency = None
        if 'frequency' in group:
            freq_grp = group['frequency']
            unit = freq_grp['unit'][()]
            unit = unit.decode('utf-8') if isinstance(unit, bytes) else unit
            
            if 'f_scaled' in freq_grp:
                f_scaled_arr = freq_grp['f_scaled'][()]
                frequency = Frequency.from_f(f=f_scaled_arr, unit=unit)
            else:
                # Basic Multiplier Dict fallback
                MULTIPLIER_DICT = {'hz': 1, 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9, 'thz': 1e12}
                f_arr = freq_grp['f'][()]
                div = MULTIPLIER_DICT.get(unit.lower(), 1)
                frequency = Frequency.from_f(f_arr / div, unit=unit)

        features = json.loads(group["features"][()]) if 'features' in group else None
        
        solver_kwargs = None
        if 'solver_kwargs' in group:
            data = group['solver_kwargs'][()]
            # Handle variable length string dataset vs scalar
            if isinstance(data, np.ndarray): data = data[0]
            solver_kwargs = jsonpickle.decode(data)

        fitter_kwargs = None
        if 'fitter_kwargs' in group:
            data = group['fitter_kwargs'][()]
            if isinstance(data, np.ndarray): data = data[0]
            fitter_kwargs = jsonpickle.decode(data)
            
        return FitSettings(frequency, features, fitter_kwargs, solver_kwargs)

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
    from pmrf.fitting.frequentist import FrequentistFitter
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
    from pmrf.fitting.bayesian import BayesianFitter
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
    
def fit(model: Model, measured: str | skrf.Network | NetworkCollection, **kwargs) -> Model:
    """Fits a model to measured data.

    This is an alternative API to ParamRF's :mod:`fitting <pmrf.fitting>` module.
    See the :func:`fit <pmrf.fitting.BaseFitter.fit>` method for more details.
    
    Internally, a fitter is first created using :func:`pmrf.fitting.Fitter`, and then :func:`fitter.fit(...)` is called,
    with the fitted model being returned. Fit results are stored in the model's metadata with key 'fit_results'.

    Key-word arguments are split into 'init' and 'fit' key-word arguments appropriately.

    Parameters
    ----------
    model: prf.Model
        The model to fit.
    measured : prf.NetworkCollection | skrf.Network | str
        The measured data.
    **kwargs
        Additional arguments forwarded to :func:`pmrf.fitting.Fitter` and :func:`pmrf.fitting.BaseFitter.fit`.

    Returns
    -------
    Model
        The fitted model.
    """        
    from pmrf.fitting import Fitter, FITTER_INIT_PARAMS
    init_kwargs = {k: kwargs.pop(k) for k in FITTER_INIT_PARAMS if k in kwargs}
    return Fitter(model, **init_kwargs).fit(measured, **kwargs).fitted_model

def fit_submodels(model: Model, measured: str | skrf.Network | NetworkCollection, **kwargs) -> Model:
    """Fits a model's submodels to measured data.

    This is an alternative API to ParamRF's :mod:`fitting <pmrf.fitting>` module.
    See the :func:`fit_submodels <pmrf.fitting.BaseFitter.fit_submodels>` method for more details.
    
    Internally, a fitter is first created using :func:`pmrf.fitting.Fitter`, and then :func:`fitter.fit_submodels(...)` is called,
    with the fitted model being returned. Fit results are stored in the model's metadata with key 'fit_results'.

    Key-word arguments are split into 'init' and 'fit' key-word arguments appropriately.

    Parameters
    ----------
    model: prf.Model
        The model to fit.    
    measured : prf.NetworkCollection | skrf.Network | str
        The measured data.
    **kwargs
        Additional arguments forwarded to :func:`pmrf.fitting.Fitter` and :func:`pmrf.fitting.BaseFitter.fit_submodels`.

    Returns
    -------
    Model
        The fitted model.
    """         
    from pmrf.fitting import Fitter, FITTER_INIT_PARAMS
    init_kwargs = {k: kwargs.pop(k) for k in FITTER_INIT_PARAMS if k in kwargs}
    return Fitter(model, **init_kwargs).fit_submodels(measured, **kwargs).fitted_model