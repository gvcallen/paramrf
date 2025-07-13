import numpy as np
from typing import Any
import jsonpickle
import h5py

from pmrf.fitting._frequentist import FrequentistFitter, FrequentistResults

class ScipyMinimizeResults(FrequentistResults):
    def encode_solver_results(self, grp: h5py.Group):
        for key, val in self.solver_results.items():
            if isinstance(val, (int, float, str, np.number)):
                grp.attrs[key] = val
            elif isinstance(val, np.ndarray):
                grp.create_dataset(key, data=val)
            elif val is None:
                grp.attrs[key] = "None"
            else:
                grp.attrs[key] = str(val)  # fallback for e.g. status messages
    
    @classmethod
    def decode_solver_results(cls, grp: h5py.Group) -> Any:
        result_dict = dict(grp.attrs)
        for key in grp:
            result_dict[key] = grp[key][()]
            
        from scipy.optimize import OptimizeResult
        return OptimizeResult(result_dict)