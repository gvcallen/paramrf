from typing import Any
import h5py
from pmrf.fitting.results import BayesianResults

class NumPyroResults(BayesianResults):
    def encode_solver_results(self, group: h5py.Group):
        samples = self.solver_results
        group['samples'] = samples
        
    @classmethod
    def decode_solver_results(cls, group: h5py.Group) -> Any:
        group['samples']