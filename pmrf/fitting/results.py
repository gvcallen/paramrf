from dataclasses import dataclass
import logging

import json
import skrf
import numpy as np
import jax.numpy as jnp
import h5py

from pmrf.network_collection import NetworkCollection
from pmrf.models.model import Model
from pmrf.results import BaseResults

@dataclass
class FitResults(BaseResults):
    """Container for the results of a model fitting process."""
    measured: skrf.Network | NetworkCollection | None = None
    fitted_model: Model | None = None

    # --------------------------------------------------------------------------
    # Subclass specific Plotting & I/O
    # --------------------------------------------------------------------------
    def _get_plot_data(self, use_initial_model=False, **kwargs) -> list[tuple]:
        model = self.initial_model if use_initial_model else self.fitted_model
        if not self.measured or not model:
            logging.warning("Missing measured data or model for plotting.")
            return []

        networks_ref = self.measured if isinstance(self.measured, NetworkCollection) else [self.measured]
        plot_data = []
        
        for meas_nw in networks_ref:
            sub_model = getattr(model, meas_nw.name) if isinstance(self.measured, NetworkCollection) else model
            try:
                fit_nw = sub_model.to_skrf(self.frequency)
                plot_data.append((
                    meas_nw.name or "Main", 
                    [meas_nw, fit_nw], 
                    [{'label': 'Measured', 'linestyle': '--', 'color': 'k'}, {'label': 'Model'}]
                ))
            except Exception as e:
                logging.warning(f"Failed to generate fitted network for {meas_nw.name}: {e}")
                
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