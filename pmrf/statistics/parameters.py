import importlib
import re
import os

import numpy as np
import pandas as pd
from scipy.stats import qmc, uniform
import uuid

from pmrf.statistics.distribution import scipy_to_string, string_to_scipy


class ParameterSet(pd.DataFrame):
    """
    A set or list of parameters (values, scales etc.) and utility functions.
    Intended to be used as a higher-level organization scheme for a large number of parameters.
    Also represents the "distribution" of the parameters into the table itself.

    In general, the following columns are present:
        - 'name': the name of a parameter
        - 'value': the un-scaled value of a parameter
        - 'scale': the scalar which should multiply with the parameter's value to yield its true value
        - 'fixed': whether or not the parameter should be viewed fixed
        - 'dist': a value of class type <scipy.rv_frozen> (which can be passed as a string e.g. "norm(0.0, 1.0)" and will be saved to file as such)
    
    NB: (for lower-level users e.g. the NetworkFitter sub-classes themselves)
    This class inherits from a pandas DataFrame. However, DataFrame read/write access is slow.
    In order to solve this, a "value cache" is provided that allows the use of fast,
    repetitive reading and writing, for values with the "fixed" flag set to False.
    To enable the cache, call "enable_cache()", and when repetitive access is finished, call "flush_cache()".
    Note that only class methods should be used while the cache is access, as opposed to pandas 'loc' and 'iloc'.
    """
    def __init__(self, df=None, data=None, columns=None, file=None, *args, **kwargs):
        """
        Method to initialize a parameter set.

        Note that, when initiailizing with "data", using columns 'minimum' and 'maximum' is allowed,
        in which case all parameters are assumed to have uniform distributions.
        Also note that the column 'fixed' can be missing, in which case all parameters are initialized with fixed == False.
        """
        # Initialize the DataFrame
        super().__init__(*args, **kwargs)

        if df is None:
            if data is not None:
                if columns is None:
                    columns = ['name', 'value', 'scale', 'fixed', 'dist', 'loc', 'scale']
                df = pd.DataFrame(data, columns=columns)
            elif file:
                match = re.match(r"\$\{([^}]+)\}/(.+)", file)
                if match:
                    module = match.group(1)
                    filename = match.group(2)

                    file = str(importlib.resources.files(module).joinpath(filename))

                df = pd.read_csv(file)      
                df['dist'] = [string_to_scipy(s) if isinstance(s, str) else None for s in df['dist']]
            # Set index
            df.set_index('name', inplace=True)                

        # Populate optional columns
        if not 'fixed' in df.columns:
            df['fixed'] = False
        if not 'scale' in df.columns:
            df['scale'] = 1.0
        
        # Set distribution if other methods were used to specify it (mainly for compatibility)
        if 'prior' in df.columns or 'pdf' in df.columns:
            replace = 'prior' if 'prior' in df.columns else 'pdf'
            df['dist'] = [string_to_scipy(s) for s in df[replace]]
            df.drop(columns=[replace], inplace=True)
        elif 'minimum' in df.columns and 'maximum' in df.columns:
            df['dist'] = [uniform(minimum, maximum-minimum) for minimum, maximum in zip(df.minimum, df.maximum)]
            df.drop(columns=['minimum', 'maximum'], inplace=True)

        # If value was not passed, set it to the mean value of the distribution specified
        if not 'value' in df.columns:
            df['value'] = [dist.mean() for dist in df.dist]

        # Overwrite the current DataFrame with the loaded data
        self._update_inplace(df)        
        self._cache_enabled = False
        self._key_cache = None
        self._value_cache = None
        self._scale_cache = None
        self._dist_cache = None
        self._loc_cache = None
        self._scale_cache = None
        self._total_dict_cache = None
        
        # TODO re-support derived params
        self._has_derived_params = False
        
    def write_csv(self, filename, make_dirs=True):
        dir_path = os.path.dirname(filename)
        if dir_path:
            if make_dirs:
                os.makedirs(dir_path, exist_ok=True)
            else:
                raise Exception(f'{dir_path} does not exist')

        cache_was_enabled = self._cache_enabled
        if cache_was_enabled:
            self.flush_cache()
        # TODO write this back into the same format as it was read in (i.e. with separate tables for separate pdf types)
        self.to_csv(filename)

        if cache_was_enabled:
            self.enable_cache()

    def evaluate_param(self, param_string):
        try:
            return float(param_string)
        except ValueError:
            derived_value = self.value.loc[self.index == param_string].iloc[0]
            return self.evaluate_param(derived_value) * self.scale[self.index == param_string].iloc[0]

    def evaluate(self) -> dict:
        if self._cache_enabled:
            for key, value, scale in zip(self._key_cache, self._value_cache, self._scale_cache):
                self._total_dict_cache[key] = value * scale 
            return self._total_dict_cache
        else:
            vectorized_evaluate = np.vectorize(self.evaluate_param)
            values = vectorized_evaluate(self.value.to_numpy())
            
            return {k: v for k, v in zip(self.index, values * self.scale)}
        
    def append(self, items: dict):
        # Ensure that all keys have been passed
        columns = set(self.columns)
        if not columns.issubset(items):
            keys_excluded = columns.difference(items)
            raise Exception(f'Items with keys {keys_excluded} not passed to parameters.append()')
        

        df_append = pd.DataFrame([items])
        df_append.set_index('name', inplace=True)
        
        self._update_inplace(pd.concat([self, df_append])) 

    def values(self, free_only=True):
        if not self._cache_enabled:
            if free_only:
                return self.loc[self.fixed == False, 'value']
            else:
                vectorized_evaluate = np.vectorize(self.evaluate_param)
                return vectorized_evaluate(self.value.to_numpy())
        else:
            if free_only:
                return self._value_cache
            else:
                raise Exception("Cannot get fixed values while ParameterSet cache is active")

    
    def update_values(self, theta: np.ndarray):
        if not self._cache_enabled:
            self.loc[self.fixed == False, 'value'] = theta
        else:
            self._value_cache = theta
            
    def enable_cache(self):
        if self._has_derived_params:
            raise Exception('Cannot enable cache when derived parameters are being used')
        self._key_cache = self.index[self.fixed == False].to_numpy()
        self._value_cache = self.loc[self.fixed == False].value.to_numpy()
        self._scale_cache = self.loc[self.fixed == False].scale.to_numpy()
        self._dist_cache = self.loc[self.fixed == False].dist.to_list()
        self._total_dict_cache = self.evaluate()
        self._cache_enabled = True

    def flush_cache(self):
        if not self._cache_enabled:
            return
        self.loc[self.fixed == False, 'value'] = self._value_cache
        self._cache_enabled = False

    def dists(self, free_only=True):
        if not self._cache_enabled:
            return self.loc[self.fixed == (not free_only), 'dist']
        else:
            if not free_only:
                raise Exception("Cannot get dists for fixed parameters while ParameterSet cache is active")
            return self._dist_cache
        
    def generate_samples(self, N=100, method='lhs', free_only=True):
        num_samples = N
        num_dimensions = len(self.names_free)

        # Create a Latin Hypercube sampler
        sampler = qmc.LatinHypercube(d=num_dimensions)

        # Generate samples
        samples = sampler.random(n=num_samples)

        # If you want to scale the sample to a specific range, for example [a, b] in each dimension
        
        if free_only:
            lower_bounds = self.min[self.fixed == False]
            upper_bounds = self.max[self.fixed == False]
        else:
            lower_bounds = self.min
            upper_bounds = self.max
        
        scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)
        
        return scaled_samples

    @property        
    def names_free(self):
        return self.index[self.fixed == False].to_list()

    @property
    def _constructor(self):
        return ParameterSet
    
    @property
    def _constructor_sliced(self):
        return pd.Series