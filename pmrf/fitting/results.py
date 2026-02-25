from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import json
import skrf
import h5py
import jsonpickle

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf._util import load_class_from_string
from pmrf.frequency import Frequency
from pmrf.network_collection import NetworkCollection
from pmrf.constants import FeatureT

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
    solver_results: Any = None
    settings: FitSettings | None = None

    # --------------------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------------------
    def plot_s_db(self, **kwargs):
        return self.plot_feature(feature='s_db', **kwargs)
    
    def plot_s_deg(self, **kwargs):
        return self.plot_feature(feature='s_deg', **kwargs)
    
    def plot_s_re(self, **kwargs):
        return self.plot_feature(feature='s_re', **kwargs)
    
    def plot_s_im(self, **kwargs):
        return self.plot_feature(feature='s_im', **kwargs)
    
    def plot_feature(self, feature='s_db', use_initial_model=False):
        """
        Plots a feature (e.g. S-parameter magnitude in dB) of the Measured vs Fitted data.
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
        feature_str = f'plot_{feature}'
        for row_idx, (label, meas, fit) in enumerate(data_to_plot):
            n_ports = meas.number_of_ports
            plot_col_idx = 0
            
            for i in range(n_ports):
                for j in range(n_ports):
                    ax = axes[row_idx, plot_col_idx]              
                    
                    if i < fit.number_of_ports and j < fit.number_of_ports:
                        func = getattr(fit, feature_str)
                        func(m=i, n=j, ax=ax, label="Model")

                    func = getattr(meas, feature_str)
                    func(m=i, n=j, ax=ax, label="Measured", linestyle='--', color='k')
                    
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
        return fig, axes

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