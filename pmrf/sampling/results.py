from dataclasses import dataclass
import logging
from typing import Any
import json

import h5py
import jsonpickle
import numpy as np
import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.models.model import Model
from pmrf.constants import FeatureT
from pmrf.util import load_class_from_string


@dataclass
class SampleSettings:
    """
    Configuration settings for the sampling process.

    Attributes
    ----------
    frequency : Frequency or None
        The frequency grid used for the feature extraction.
    features : list of FeatureT or None
        The list of features extracted during sampling.
    """
    frequency: Frequency | None = None
    features: list[FeatureT] | None = None


@dataclass
class SampleResults:
    """
    Container for the results of a model sampling process.
    
    Attributes
    ----------
    initial_model : Model or None
        The model with the initial parameters.
    sampled_models : list[Model] or None
        The sampled models.
    sampled_params : jnp.ndarray or None
        The raw parameter samples.
    sampled_features : jnp.ndarray or None
        The extracted features for the samples.
    backend_results : Any
        The raw result object returned by the sampling backend/algorithm.
    settings : SampleSettings or None
        The configuration used to execute the sampling.
    """    
    initial_model: Model | None = None
    sampled_models: list[Model] | None = None
    sampled_params: jnp.ndarray | None = None
    sampled_features: jnp.ndarray | None = None
    backend_results: Any = None
    settings: SampleSettings | None = None

    # --------------------------------------------------------------------------
    # Public HDF5 IO Interface
    # --------------------------------------------------------------------------

    def save_hdf(self, path: str, metadata: dict | None = None):
        """
        Save the sampling results to an HDF5 file.
        """
        with h5py.File(path, 'w') as f:
            self._write_to_group(f, metadata)

    @classmethod
    def load_hdf(cls, path: str) -> "SampleResults":
        """
        Load sampling results from an HDF5 file.
        """
        with h5py.File(path, 'r') as f:
            return cls._read_from_group(f)

    # --------------------------------------------------------------------------
    # Polymorphic Backend Results IO
    # --------------------------------------------------------------------------

    def encode_backend_results(self, group: h5py.Group):
        """
        Encode raw backend results into an HDF5 group.
        """
        if self.backend_results is None:
            return

        if isinstance(self.backend_results, (dict, list, tuple, set, SampleResults)):
            self._encode_recursive(self.backend_results, group)
        else:
            try:
                data = jsonpickle.encode(self.backend_results)
                dt = h5py.special_dtype(vlen=str)
                group.create_dataset('data', data=np.array([data], dtype=dt))
            except Exception as e:
                logging.error(f"Failed to encode backend results: {e}")

    @classmethod
    def decode_backend_results(cls, group: h5py.Group) -> Any:
        """
        Decode backend results from an HDF5 group.
        """
        if '__type__' in group.attrs:
            return cls._decode_recursive(group)
        elif 'data' in group:
            try:
                data = group['data'][()]
                if isinstance(data, np.ndarray): data = data[0]
                if isinstance(data, bytes): data = data.decode('utf-8')
                return jsonpickle.decode(data)
            except Exception as e:
                logging.error(f"Failed to decode backend results: {e}")
                return None
        return None

    # --------------------------------------------------------------------------
    # Recursive Encoding Helpers (Protected)
    # --------------------------------------------------------------------------

    def _encode_recursive(self, obj: Any, group: h5py.Group):
        if isinstance(obj, SampleResults):
            group.attrs['__type__'] = 'SampleResults'
            obj._write_to_group(group)
            return

        if isinstance(obj, dict):
            group.attrs['__type__'] = 'dict'
            for k, v in obj.items():
                sub_grp = group.create_group(str(k))
                self._encode_recursive(v, sub_grp)
            return

        if isinstance(obj, (list, tuple, set)):
            group.attrs['__type__'] = type(obj).__name__ 
            for i, item in enumerate(obj):
                sub_grp = group.create_group(str(i))
                self._encode_recursive(item, sub_grp)
            return

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

        if type_tag == 'SampleResults':
            return SampleResults._read_from_group(group)

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

        else: 
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
        internal_grp['sample_results_cls'] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        internal_grp['version'] = 1
        
        if metadata:
            self._save_dict_to_group(metadata, metadata_grp)

        # 2. Models
        if self.initial_model: 
            self.initial_model.write_hdf(group.create_group('initial_model'))
            
        if self.sampled_models:
            sm_grp = group.create_group('sampled_models')
            for i, model in enumerate(self.sampled_models):
                model.write_hdf(sm_grp.create_group(str(i)))

        # 3. JAX Arrays (cast to NumPy for HDF5 compatibility)
        if self.sampled_params is not None:
            group.create_dataset('sampled_params', data=np.asarray(self.sampled_params))
            
        if self.sampled_features is not None:
            group.create_dataset('sampled_features', data=np.asarray(self.sampled_features))

        # 4. Settings
        if self.settings:
            self._write_settings(group.create_group('settings'))

        # 5. Backend Results
        if self.backend_results is not None:
            self.encode_backend_results(group.create_group('backend_results'))

    @classmethod
    def _read_from_group(cls, group: h5py.Group) -> "SampleResults":
        """Internal driver to load full object state."""
        
        # 1. Determine Class Type
        target_cls = cls
        if 'metadata' in group:
            meta = group['metadata'].get('__pmrf__', {})
            if 'sample_results_cls' in meta:
                cls_path = meta['sample_results_cls'][()]
                cls_path = cls_path.decode('utf-8') if isinstance(cls_path, bytes) else cls_path
                target_cls = load_class_from_string(cls_path)

        # 2. Load Models
        initial_model = Model.read_hdf(group['initial_model']) if 'initial_model' in group else None
        
        sampled_models = None
        if 'sampled_models' in group:
            sm_grp = group['sampled_models']
            keys = sorted(sm_grp.keys(), key=lambda x: int(x))
            sampled_models = [Model.read_hdf(sm_grp[k]) for k in keys]

        # 3. Load Arrays (cast back to JAX arrays)
        sampled_params = jnp.array(group['sampled_params'][()]) if 'sampled_params' in group else None
        sampled_features = jnp.array(group['sampled_features'][()]) if 'sampled_features' in group else None

        # 4. Load Settings & Backend Results
        settings = cls._read_settings(group['settings']) if 'settings' in group else None
        backend_results = target_cls.decode_backend_results(group['backend_results']) if 'backend_results' in group else None

        return target_cls(
            initial_model=initial_model,
            sampled_models=sampled_models,
            sampled_params=sampled_params,
            sampled_features=sampled_features,
            backend_results=backend_results,
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
                SampleResults._save_dict_to_group(v, subgrp)
            else:
                group[k] = json.dumps(v)

    def _write_settings(self, group: h5py.Group):
        if self.settings.frequency is not None:
            frequency_grp = group.create_group('frequency')
            frequency_grp['f'] = self.settings.frequency.f
            frequency_grp['f_scaled'] = self.settings.frequency.f_scaled
            frequency_grp['unit'] = self.settings.frequency.unit
        
        if self.settings.features is not None:
            group.create_dataset('features', data=json.dumps(self.settings.features))

    @staticmethod
    def _read_settings(group: h5py.Group):
        frequency = None
        if 'frequency' in group:
            freq_grp = group['frequency']
            unit = freq_grp['unit'][()]
            unit = unit.decode('utf-8') if isinstance(unit, bytes) else unit
            
            if 'f_scaled' in freq_grp:
                f_scaled_arr = freq_grp['f_scaled'][()]
                frequency = Frequency.from_f(f=f_scaled_arr, unit=unit)
            else:
                MULTIPLIER_DICT = {'hz': 1, 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9, 'thz': 1e12}
                f_arr = freq_grp['f'][()]
                div = MULTIPLIER_DICT.get(unit.lower(), 1)
                frequency = Frequency.from_f(f_arr / div, unit=unit)

        features = json.loads(group["features"][()]) if 'features' in group else None
        
        return SampleSettings(frequency, features)