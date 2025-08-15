from typing import Any
import io   
import h5py

from pmrf.fitting._bayesian import BayesianResults

class AnestheticResults(BayesianResults):
    from anesthetic import NestedSamples
    
    @property
    def nested_samples(self) -> NestedSamples:
        return self.solver_results    
    
    def samples(self):
        nested_samples = self.nested_samples
        columns = nested_samples.columns
        param_names = [columns[i][0] for i in range(len(columns))]
        param_names = [name for name in param_names if name not in {'logL', 'logL_birth', 'nlive'}]
        return nested_samples.loc[:, param_names]
    
    def prior_samples(self):
        nested_samples = self.nested_samples.prior()
        columns = nested_samples.columns
        param_names = [columns[i][0] for i in range(len(columns))]
        param_names = [name for name in param_names if name not in {'logL', 'logL_birth', 'nlive'}]
        return nested_samples.loc[:, param_names]
    
    def weights(self):
        nested_samples = self.nested_samples
        return nested_samples.get_weights()
    
    def prior_weights(self):
        nested_samples = self.nested_samples
        return nested_samples.prior().get_weights()
    
    def encode_solver_results(self, group: h5py.Group):
        samples = self.solver_results
        group['samples'] = samples.to_csv()
        
    @classmethod
    def decode_solver_results(cls, group: h5py.Group) -> Any:
        from anesthetic import NestedSamples, read_csv
        import pandas as pd
        
        csv_str = group['samples'][()]
        csv_str = csv_str.decode('utf-8') if isinstance(csv_str, bytes) else csv_str
        samples = NestedSamples(read_csv(io.StringIO(csv_str)))
        # samples = NestedSamples(pd.read_csv(io.StringIO(csv_str), index_col=0))
        return samples    