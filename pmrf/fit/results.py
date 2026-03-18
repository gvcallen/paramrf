from dataclasses import dataclass
import logging

import json
import skrf
import numpy as np
import jax.numpy as jnp
import h5py

from pmrf.network_collection import NetworkCollection
from pmrf.core import Model
from pmrf.results import BaseResults
from pmrf.features import extract_features

@dataclass
class FitResults(BaseResults):
    """Container for the results of a model fitting process."""
    measured: skrf.Network | NetworkCollection | None = None
    fitted_model: Model | None = None

    # --------------------------------------------------------------------------
    # Subclass specific Plotting & I/O
    # --------------------------------------------------------------------------
    def _get_feature_plot_data(self, feature: str, use_initial_model=False, **kwargs) -> list[dict]:
        """
        Extracts the requested feature from both the measured data and the 
        fitted model to visualize the quality of the fit.
        """
        model = self.initial_model if use_initial_model else self.fitted_model
        if not self.measured or not model or not self.frequency:
            logging.warning("Missing measured data, model, or frequency for plotting.")
            return []

        plot_data = []
        networks_ref = self.measured if isinstance(self.measured, NetworkCollection) else [self.measured]

        try:
            # We iterate over networks so that if the user fits a NetworkCollection, 
            # we plot the feature for every network/sub-model pair in the collection.
            for nw in networks_ref:
                # If it's a collection, target the specific network and sub-model by name
                feature_spec = {nw.name: feature} if isinstance(self.measured, NetworkCollection) else [feature]
                
                # 1. Extract from Measured Data
                y_meas = extract_features(self.measured, self.frequency, feature_spec)
                y_meas = np.real(y_meas[:, 0]) # Cast to real to strip complex dtype
                
                label_suffix = f" ({nw.name})" if nw.name else ""
                
                plot_data.append({
                    'y': y_meas,
                    'label': f'Measured{label_suffix}',
                    'linestyle': '--',
                    'color': 'k',
                    'linewidth': 1.5
                })
                
                # 2. Extract from Model
                y_mod = extract_features(model, self.frequency, feature_spec)
                y_mod = np.real(y_mod[:, 0])
                
                plot_data.append({
                    'y': y_mod,
                    'label': f'Model{label_suffix}',
                    'linestyle': '-',
                    'linewidth': 1.5
                })
                
        except Exception as e:
            logging.warning(f"Failed to extract '{feature}' for fitting plot: {e}")

        return plot_data

    def _write_data(self, f: h5py.File):
        if self.fitted_model:
            self._write_model(f.create_group('fitted_model'), self.fitted_model)

        if self.measured:
            meas_grp = f.create_group('measured')
            networks = self.measured if isinstance(self.measured, NetworkCollection) else [self.measured]
            if isinstance(self.measured, NetworkCollection) and self.measured.params:
                meas_grp.attrs['collection_params'] = json.dumps(self.measured.params)
            for ntwk in networks:
                self._write_network(meas_grp.create_group(ntwk.name or 'network'), ntwk)

    @classmethod
    def _read_data(cls, f: h5py.File, kwargs: dict):
        if 'models' in f and 'fitted' in f['models']:
            kwargs['fitted_model'] = cls._read_model(f['models/fitted'])

        if 'measured' in f:
            meas_grp = f['measured']
            networks = [cls._read_network(meas_grp[k]) for k in meas_grp.keys()]
            if len(networks) == 1 and 'collection_params' not in meas_grp.attrs:
                kwargs['measured'] = networks[0]
            else:
                params = json.loads(cls._decode_str(meas_grp.attrs['collection_params'])) if 'collection_params' in meas_grp.attrs else None
                nc = NetworkCollection(params=params)
                for nw in networks: nc.add(nw)
                kwargs['measured'] = nc