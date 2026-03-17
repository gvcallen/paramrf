from dataclasses import dataclass
import logging

import numpy as np
import jax.numpy as jnp
import h5py

from pmrf.models.model import Model
from pmrf.results import BaseResults
from pmrf.features import extract_features

@dataclass
class SampleResults(BaseResults):
    """Container for the results of a model sampling process."""
    sampled_models: list[Model] | None = None
    sampled_params: jnp.ndarray | None = None
    sampled_features: jnp.ndarray | None = None
    
    # --------------------------------------------------------------------------
    # Subclass specific Plotting & I/O
    # --------------------------------------------------------------------------
    def _get_feature_plot_data(self, feature: str, plot_samples=10, **kwargs) -> list[dict]:
        """
        Extracts the requested feature from the initial model and a subset 
        of the sampled posterior models to visualize variance.
        """
        if not self.initial_model or not self.frequency:
            logging.warning("Missing initial model or frequency for plotting.")
            return []

        plot_data = []

        # 1. Plot the initial model (Base case)
        try:
            y_init = extract_features(self.initial_model, self.frequency, [feature])
            y_init = np.real(y_init[:, 0]) # Cast to real to strip complex dtype
            
            plot_data.append({
                'y': y_init,
                'label': 'Initial Model',
                'linestyle': '-',
                'color': 'k',
                'linewidth': 2,
                'zorder': 10 # Keep the initial model on top of the samples
            })
        except Exception as e:
            logging.warning(f"Failed to extract '{feature}' from initial model: {e}")

        # 2. Plot the sampled models (Ghost traces)
        if self.sampled_models:
            step = max(1, len(self.sampled_models) // plot_samples)
            for i, sm in enumerate(self.sampled_models[::step]):
                try:
                    y_sm = extract_features(sm, self.frequency, [feature])
                    y_sm = np.real(y_sm[:, 0])
                    
                    plot_data.append({
                        'y': y_sm,
                        'label': 'Sample' if i == 0 else None, # Only label the first one
                        'linestyle': '-',
                        'color': 'C0',
                        'alpha': 0.3, # Make samples semi-transparent
                        'linewidth': 1
                    })
                except Exception:
                    pass

        return plot_data

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