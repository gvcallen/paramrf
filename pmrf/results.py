"""
The base results class for results from runners such as fitters and samplers.
"""
from dataclasses import dataclass
import importlib.metadata
import logging
from typing import Any
import json
import io

import numpy as np
import skrf
import h5py
import jsonpickle

from pmrf.models.model import Model
from pmrf.frequency import Frequency
from pmrf.util import load_class_from_string
from pmrf.constants import FeatureT


@dataclass
class BaseResults:
    """
    Base container for the results of pmrf processes (Fitting, Sampling, etc.)
    """
    initial_model: Model | None = None
    frequency: Frequency | None = None
    features: list[FeatureT] | None = None
    backend_results: Any = None
    backend_class: str | None = None     # e.g., "pmrf.fitting.scipy.SciPyMinimizeFitter"
    run_kwargs: dict | None = None       # Arguments passed to .run()

    # --------------------------------------------------------------------------
    # Shared Plotting Utilities
    # --------------------------------------------------------------------------
    def __getattr__(self, name: str):
        """
        Intercepts calls like `results.plot_s11_db()` dynamically.
        Extracts the feature name ('s11_db') and routes it to the generic plotter.
        """
        if name.startswith('plot_'):
            feature_name = name[5:] # Strip the 'plot_' prefix
            return lambda **kwargs: self._plot_single_feature(feature_name, **kwargs)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def _plot_single_feature(self, feature: str, **kwargs):
        """
        Plots a specific feature using data provided by the subclass.
        """
        import matplotlib.pyplot as plt

        if not self.frequency:
            logging.warning("Missing frequency settings for plotting.")
            return

        # Subclasses provide a list of dictionaries containing 'y' data and plot kwargs
        plot_data = self._get_feature_plot_data(feature, **kwargs)
        if not plot_data:
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        
        # 1. Plot against the correctly scaled frequency!
        x = self.frequency.f_scaled 

        for data_dict in plot_data:
            y = data_dict.pop('y')
            ax.plot(x, y, **data_dict)

        # 2. Format the graph
        ax.set_title(f"Optimization Feature: {feature}")
        ax.set_xlabel(f"Frequency ({self.frequency.unit})")
        
        # Try to infer a nice Y-axis label
        if str(feature).endswith('_db'):
            ax.set_ylabel("Magnitude (dB)")
        elif str(feature).endswith('_deg'):
            ax.set_ylabel("Phase (Degrees)")
        elif str(feature).endswith('_mag'):
            ax.set_ylabel("Magnitude (Linear)")
            
        ax.grid(True, alpha=0.5)
        ax.legend(fontsize='small')
        fig.tight_layout()
        
        return fig, ax

    def _get_feature_plot_data(self, feature: str, **kwargs) -> list[dict]:
        """Override in subclasses to provide y-data and plot kwargs."""
        return []

    # --------------------------------------------------------------------------
    # Public HDF5 IO Interface
    # --------------------------------------------------------------------------
    def save_hdf(self, path: str, metadata: dict | None = None):
        with h5py.File(path, 'w') as f:
            # 1. Metadata
            f.attrs['pmrf_class'] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
            try:
                f.attrs['pmrf_version'] = importlib.metadata.version("pmrf")
            except importlib.metadata.PackageNotFoundError:
                f.attrs['pmrf_version'] = "unknown (not installed)"
            if metadata:
                f.attrs['user_metadata'] = json.dumps(metadata)

            # 2. Config & Settings
            if self.frequency:
                f.create_dataset('f_scaled', data=self.frequency.f_scaled)
                f.attrs['freq_unit'] = self.frequency.unit
            if self.features: 
                f.attrs['features'] = json.dumps(self.features)
            if self.run_kwargs: 
                f.attrs['run_kwargs'] = jsonpickle.encode(self.run_kwargs)
            
            # 3. Models
            if self.initial_model: 
                self._write_model(f.create_group('initial_model'), self.initial_model)

            # 4. Subclass Data Hook (Fitted models, sampled arrays, measured data)
            self._write_data(f)

            # 5. Backend Results (Streamed via the Fitter/Sampler class)
            if self.backend_results is not None:
                if not self.backend_class:
                    logging.warning("backend_results present but backend_class not specified. Skipping backend serialization.")
                else:
                    f.attrs['backend_class'] = self.backend_class
                    backend_cls = load_class_from_string(self.backend_class)
                    
                    stream = io.BytesIO()
                    backend_cls.write_results(stream, self.backend_results)
                    f.create_dataset('backend_results', data=np.void(stream.getvalue()))

    @classmethod
    def load_hdf(cls, path: str) -> "BaseResults":
        with h5py.File(path, 'r') as f:
            cls_path = cls._decode_str(f.attrs.get('pmrf_class', f"{cls.__module__}.{cls.__qualname__}"))
            target_cls = load_class_from_string(cls_path)
            kwargs = {}

            # Load Config
            if 'f_scaled' in f:
                kwargs['frequency'] = Frequency.from_f(f=f['f_scaled'][()], unit=cls._decode_str(f.attrs['freq_unit']))
            if 'features' in f.attrs: 
                kwargs['features'] = json.loads(cls._decode_str(f.attrs['features']))
            if 'run_kwargs' in f.attrs: 
                kwargs['run_kwargs'] = jsonpickle.decode(cls._decode_str(f.attrs['run_kwargs']))

            # Load Base Models
            if 'initial_model' in f:
                kwargs['initial_model'] = cls._read_model(f['initial_model'])

            # Load Subclass Data
            target_cls._read_data(f, kwargs)

            # Load Backend Results
            if 'backend_results' in f and 'backend_class' in f.attrs:
                backend_class_str = cls._decode_str(f.attrs['backend_class'])
                kwargs['backend_class'] = backend_class_str
                backend_cls = load_class_from_string(backend_class_str)
                
                raw_bytes = f['backend_results'][()].tobytes()
                stream = io.BytesIO(raw_bytes)
                kwargs['backend_results'] = backend_cls.read_results(stream)

            return target_cls(**kwargs)

    # --------------------------------------------------------------------------
    # Subclass Hooks (To be overridden)
    # --------------------------------------------------------------------------
    def _write_data(self, f: h5py.File): pass
    @classmethod
    def _read_data(cls, f: h5py.File, kwargs: dict): pass

    # --------------------------------------------------------------------------
    # Internal I/O Utilities
    # --------------------------------------------------------------------------
    @staticmethod
    def _decode_str(data: Any) -> str:
        if isinstance(data, np.ndarray): data = data[0]
        return data.decode('utf-8') if isinstance(data, bytes) else str(data)

    @staticmethod
    def _write_model(group: h5py.Group, model: Model):
        params_grp = group.create_group('params')
        for name, param in model.named_params().items():
            params_grp.create_dataset(name, data=param.to_json())
        group.create_dataset('raw', data=jsonpickle.encode(model))

    @staticmethod
    def _read_model(group: h5py.Group) -> Model:
        return jsonpickle.decode(BaseResults._decode_str(group['raw'][()]))

    @staticmethod
    def _write_network(group: h5py.Group, ntwk: skrf.Network):
        group.attrs['name'] = ntwk.name or 'network'
        if isinstance(ntwk.z0, (int, float, complex)) or ntwk.z0.size == 1:
            group.attrs['z0_scalar'] = np.ravel(ntwk.z0)[0]
        else:
            group.create_dataset('z0', data=ntwk.z0)
        group.create_dataset('s', data=ntwk.s)
        group.create_dataset('f', data=ntwk.f)
        if ntwk.params: group.attrs['params'] = json.dumps(ntwk.params)

    @staticmethod
    def _read_network(group: h5py.Group) -> skrf.Network:
        name = BaseResults._decode_str(group.attrs.get('name', 'network'))
        z0 = group['z0'][()] if 'z0' in group else group.attrs['z0_scalar']
        params = json.loads(BaseResults._decode_str(group.attrs['params'])) if 'params' in group.attrs else None
        return skrf.Network(s=group['s'][()], f=group['f'][()], z0=z0, name=name, params=params)