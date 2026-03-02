from dataclasses import dataclass
import logging

import numpy as np
import jax.numpy as jnp
import h5py

from pmrf.models import Model
from pmrf.results import BaseResults

@dataclass
class SampleResults(BaseResults):
    """Container for the results of a model sampling process."""
    sampled_models: list[Model] | None = None
    sampled_params: jnp.ndarray | None = None
    sampled_features: jnp.ndarray | None = None

    # --------------------------------------------------------------------------
    # Subclass specific Plotting & I/O
    # --------------------------------------------------------------------------
    def _get_plot_data(self, plot_samples=10, **kwargs) -> list[tuple]:
        """Plots the initial model and a subset of the sampled posterior models."""
        if not self.initial_model:
            logging.warning("Missing initial model for plotting.")
            return []
            
        try:
            base_nw = self.initial_model.to_skrf(self.frequency)
        except Exception as e:
            logging.warning(f"Could not convert initial model to skrf: {e}")
            return []

        nws_to_plot = [base_nw]
        pkws_to_plot = [{'label': 'Initial Model', 'color': 'k', 'linewidth': 2}]
        
        if self.sampled_models:
            step = max(1, len(self.sampled_models) // plot_samples)
            for i, sm in enumerate(self.sampled_models[::step]):
                try:
                    nws_to_plot.append(sm.to_skrf(self.frequency))
                    pkws_to_plot.append({'label': 'Sample' if i==0 else None, 'color': 'C0', 'alpha': 0.3})
                except Exception:
                    pass
                    
        return [("Sampling Results", nws_to_plot, pkws_to_plot)]

    def _write_data(self, f: h5py.File):
        if self.sampled_params is not None:
            f.create_dataset('sampled_params', data=np.asarray(self.sampled_params))
        if self.sampled_features is not None:
            f.create_dataset('sampled_features', data=np.asarray(self.sampled_features))

        if self.sampled_models:
            sm_grp = f.create_group('sampled_models')
            for i, model in enumerate(self.sampled_models):
                self._write_model(sm_grp.create_group(str(i)), model)

    @classmethod
    def _read_data(cls, f: h5py.File, kwargs: dict):
        if 'sampled_params' in f:
            kwargs['sampled_params'] = jnp.array(f['sampled_params'][()])
        if 'sampled_features' in f:
            kwargs['sampled_features'] = jnp.array(f['sampled_features'][()])
            
        if 'models' in f and 'sampled' in f['models']:
            sm_grp = f['models/sampled']
            keys = sorted(sm_grp.keys(), key=lambda x: int(x))
            kwargs['sampled_models'] = [cls._read_model(sm_grp[k]) for k in keys]