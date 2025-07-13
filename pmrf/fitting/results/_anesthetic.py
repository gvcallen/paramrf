from typing import Any
import io   
import h5py

from pmrf.fitting._bayesian import BayesianResults

class AnestheticResults(BayesianResults):
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
    
    def plot_params(self, param_names=None, title='params', label='posterior', priors=False, fig_size=None, fig=None, ax=None, **kwargs):
        from anesthetic import make_2d_axes
        
        nested_samples = self.solver_results
        params = param_names or list(self.model.params().keys())

        if ax is None:
            fig, ax = make_2d_axes(params, figsize=fig_size)

        for i in range(ax.shape[0]):  # Loop over rows
            for j in range(ax.shape[1]):  # Loop over columns
                axi = ax.iloc[i, j]
                axi.set_ylabel(axi.get_ylabel(), rotation='horizontal')

        if priors:
            prior_samples = nested_samples.prior()
            prior_samples.plot_2d(ax, label='prior', **kwargs)
        
        nested_samples.plot_2d(ax, label=label, **kwargs)
        if priors:
            ax.iloc[-1, 0].legend(bbox_to_anchor=(len(ax)/2, len(ax)), loc='lower center', ncol=2)
        
        return fig, ax