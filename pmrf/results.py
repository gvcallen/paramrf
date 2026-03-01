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

from pmrf.models import Model
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
    def plot_s_db(self, **kwargs): return self.plot_property(property='s_db', **kwargs)
    def plot_s_deg(self, **kwargs): return self.plot_property(property='s_deg', **kwargs)
    def plot_s_re(self, **kwargs): return self.plot_property(property='s_re', **kwargs)
    def plot_s_im(self, **kwargs): return self.plot_property(property='s_im', **kwargs)

    def plot_property(self, property='s_db', **kwargs):
        """
        Plots a property (e.g. S-parameter magnitude in dB) of the results.
        Delegates the specific networks to plot to the subclass.
        """
        import matplotlib.pyplot as plt

        if not self.frequency:
            logging.warning("Missing frequency settings for plotting.")
            return

        # Subclasses provide a list of tuples: 
        # (panel_title, list_of_skrf_networks, list_of_plot_kwargs)
        plot_data = self._get_plot_data(**kwargs)
        if not plot_data: return

        n_rows = len(plot_data)
        max_ports = max(networks[0].number_of_ports for _, networks, _ in plot_data if networks)
        if max_ports == 0: return
        
        fig, axes = plt.subplots(n_rows, max_ports**2, figsize=(4 * max_ports**2, 3.5 * n_rows), squeeze=False)

        for row, (label, networks, plot_kwargs_list) in enumerate(plot_data):
            col = 0
            n_ports = networks[0].number_of_ports
            for i in range(n_ports):
                for j in range(n_ports):
                    ax = axes[row, col]
                    
                    for nw, pkwargs in zip(networks, plot_kwargs_list):
                        if i < nw.number_of_ports and j < nw.number_of_ports:
                            getattr(nw, f'plot_{property}')(m=i, n=j, ax=ax, **pkwargs)
                    
                    ax.set_title(f"{label} - S{i+1}{j+1}")
                    ax.grid(True, alpha=0.5)
                    
                    if col == 0: ax.legend(fontsize='small')
                    else: ax.get_legend().remove() if ax.get_legend() else None
                    col += 1
                    
            for k in range(col, max_ports**2): axes[row, k].axis('off')

        fig.tight_layout()
        return fig, axes

    def _get_plot_data(self, **kwargs) -> list[tuple]:
        """Override in subclasses to provide networks for `plot_property`."""
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
            cfg = f.create_group('config')
            if self.frequency:
                cfg.create_dataset('f_scaled', data=self.frequency.f_scaled)
                cfg.attrs['freq_unit'] = self.frequency.unit
            if self.features: 
                cfg.attrs['features'] = json.dumps(self.features)
            if self.run_kwargs: 
                cfg.attrs['run_kwargs'] = jsonpickle.encode(self.run_kwargs)
            
            self._write_config_extra(cfg)

            # 3. Models Base
            models_grp = f.create_group('models')
            if self.initial_model: 
                self._write_model(models_grp.create_group('initial'), self.initial_model)

            # 4. Subclass Data Hook (Fitted models, sampled arrays, measured data)
            self._write_data(f, models_grp)

            # 5. Backend Results (Streamed via the Fitter/Sampler class)
            if self.backend_results is not None:
                if not self.backend_class:
                    logging.warning("backend_results present but backend_class not specified. Skipping backend serialization.")
                else:
                    f.attrs['backend_class'] = self.backend_class
                    backend_cls = load_class_from_string(self.backend_class)
                    
                    stream = io.BytesIO()
                    backend_cls.write_results(stream, self.backend_results)
                    f.create_dataset('backend_results_bin', data=np.void(stream.getvalue()))

    @classmethod
    def load_hdf(cls, path: str) -> "BaseResults":
        with h5py.File(path, 'r') as f:
            cls_path = cls._decode_str(f.attrs.get('pmrf_class', f"{cls.__module__}.{cls.__qualname__}"))
            target_cls = load_class_from_string(cls_path)
            kwargs = {}

            # Load Config
            if 'config' in f:
                cfg = f['config']
                if 'f_scaled' in cfg:
                    kwargs['frequency'] = Frequency.from_f(f=cfg['f_scaled'][()], unit=cls._decode_str(cfg.attrs['freq_unit']))
                if 'features' in cfg.attrs: 
                    kwargs['features'] = json.loads(cls._decode_str(cfg.attrs['features']))
                if 'run_kwargs' in cfg.attrs: 
                    kwargs['run_kwargs'] = jsonpickle.decode(cls._decode_str(cfg.attrs['run_kwargs']))
                target_cls._read_config_extra(cfg, kwargs)

            # Load Base Models
            if 'models' in f and 'initial' in f['models']:
                kwargs['initial_model'] = cls._read_model(f['models/initial'])

            # Load Subclass Data
            target_cls._read_data(f, kwargs)

            # Load Backend Results
            if 'backend_results_bin' in f and 'backend_class' in f.attrs:
                backend_class_str = cls._decode_str(f.attrs['backend_class'])
                kwargs['backend_class'] = backend_class_str
                backend_cls = load_class_from_string(backend_class_str)
                
                raw_bytes = f['backend_results_bin'][()].tobytes()
                stream = io.BytesIO(raw_bytes)
                kwargs['backend_results'] = backend_cls.read_results(stream)

            return target_cls(**kwargs)

    # --------------------------------------------------------------------------
    # Subclass Hooks (To be overridden)
    # --------------------------------------------------------------------------
    def _write_config_extra(self, cfg: h5py.Group): pass
    @classmethod
    def _read_config_extra(cls, cfg: h5py.Group, kwargs: dict): pass

    def _write_data(self, f: h5py.File, models_grp: h5py.Group): pass
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
        if hasattr(model, 'named_params'):
            params_grp = group.create_group('params')
            for name, param in model.named_params.items():
                params_grp.create_dataset(name, data=param.to_json())
        group.create_dataset('raw_state', data=jsonpickle.encode(model))

    @staticmethod
    def _read_model(group: h5py.Group) -> Model:
        return jsonpickle.decode(BaseResults._decode_str(group['raw_state'][()]))

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