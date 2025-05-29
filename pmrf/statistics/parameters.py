import importlib
import re

import numpy as np
import pandas as pd
from scipy.stats import qmc
import uuid

from pmrf.statistics.pdf import UniformPDF, from_string

class ParameterSet(pd.DataFrame):
    """
    A set or list of parameters (values, scales etc.) and utility functions.
    Intended to be used as a higher-level organization scheme for a large number of parameters.
    Also wraps the "pdfs" into the table itself.

    In general, the following columns should be present:
        - 'name': the name of a parameter
        - 'value': the un-scaled value of a parameter
        - 'scale': the scale which should multiply with the parameter's value to yield its true value
        - 'fixed': whether or not the parameter should be fixed
        - 'pdf': a value of type <paramrf.statistics.pdf>. To simply bound the value with a min/max, use <paramrf.statistics.Uniform>.
    
    NB: (for lower-level users e.g. the CircuitFitter sub-classes themselves)
    This class inherits from a pandas DataFrame. However, DataFrame read/write access is slow.
    In order to solve this, a "value cache" is provided that allows the use of fast,
    repetitive reading and writing, for values with the "fixed" flag set to False.
    To enable the cache, call "enable_cache()", and when repetitive access is finished, call "flush_cache()".
    Note that only class methods should be used while the cache is access, as opposed to pandas 'loc' and 'iloc'.
    """
    def __init__(self, df=None, data=None, columns=None, file=None, *args, **kwargs):
        """
        Method to initialize a parameter set.

        Note that, when initiailizing with "data", using columns 'minimum' and 'maximum' is allowed, in which case uniform pdfs are created.
        Also note that the column 'fixed' can be missing, in which case all parameters are initialized with fixed == False.
        """
        # Initialize the DataFrame
        super().__init__(*args, **kwargs)

        if data is not None:
            if columns is None:
                columns = ['name', 'value', 'scale', 'fixed', 'pdf']
            df = pd.DataFrame(data, columns=columns)
            
            if not 'fixed' in df.columns:
                df['fixed'] = False
            
            if not 'scale' in df.columns:
                df['scale'] = 1.0

            # For compatibility
            if 'prior' in df.columns:
                df['pdf'] = df['prior']
                df.drop(columns=['prior'], inplace=True)

            if not 'value' in df.columns:
                df['value'] = [pdf.mean for pdf in df.pdf]
            
            # For compatibility
            if 'minimum' in df.columns and 'maximum' in df.columns:
                df['pdf'] = [UniformPDF(minimum, maximum) for minimum, maximum in zip(df.minimum, df.maximum)]
                df.drop(columns=['minimum', 'maximum'], inplace=True)
        elif file:
            # Use re.sub to replace the patterns
            match = re.match(r"\$\{([^}]+)\}/(.+)", file)
            if match:
                module = match.group(1)
                filename = match.group(2)

                file = str(importlib.resources.files(module).joinpath(filename))

            df = pd.read_csv(file)

            dists = []
            for dist_string, fixed in zip(df.pdf, df.fixed):
                try:
                    dists.append(from_string(dist_string))
                except:
                    if not fixed:
                        raise Exception("Derived parameters must have Fixed == True")
                    dists.append(dist_string)

            df['pdf'] = dists
        
        if 'name' in df.columns:
            df.set_index('name', inplace=True)
        
        # Overwrite the current DataFrame with the loaded data
        self._update_inplace(df)
        self._cache_enabled = False
        self._key_cache = None
        self._value_cache = None
        self._scale_cache = None
        self._pdf_cache = None
        self._total_dict_cache = None
        
    def write_csv(self, filename):
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
            self._regenerate_id()
            self._value_cache = theta
            
    def get_state_id(self):
        if self._cache_enabled:
            return self._state_uuid
        else:
            return uuid.uuid4()
        
    def update_bounds(self, bounds: dict, update_values=False):
        for param, bound in bounds.items():
            self.loc[param, 'pdf'].value = bound
            if update_values:
                self.loc[param, 'value'] = (bound[1] + bound[0]) / 2.0

    def enable_cache(self):
        # Currently not working
        pass
        # self._key_cache = self.index[self.fixed == False].to_numpy()
        # self._value_cache = self.loc[self.fixed == False].value.to_numpy()
        # self._scale_cache = self.loc[self.fixed == False].scale.to_numpy()
        # self._pdf_cache = self.loc[self.fixed == False].pdf.to_list()
        # self._total_dict_cache = self.evaluate()
        # self._cache_enabled = True
        # self._regenerate_id()

    def flush_cache(self):
        # Currently not working
        pass
        # self.loc[self.fixed == False, 'value'] = self._value_cache
        # self._cache_enabled = False

    def pdfs(self, free_only=True):
        if not self._cache_enabled:
            return self.loc[self.fixed == (not free_only), 'pdf']
        else:
            if not free_only:
                raise Exception("Cannot get pdfs for fixed parameters while ParameterSet cache is active")
            return self._pdf_cache
        
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
    
    def _regenerate_id(self):
        self._state_uuid = uuid.uuid4()
        
    @property
    def min(self):
        return np.array([pdf.min for pdf in self.pdf])
    
    @property
    def max(self):
        return np.array([pdf.max for pdf in self.pdf])