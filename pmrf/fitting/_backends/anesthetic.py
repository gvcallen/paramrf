from typing import Any
import io

import h5py
import jax.numpy as jnp

from pmrf.fitting.bayesian import BayesianResults

class AnestheticResults(BayesianResults):
    """
    Results container for Nested Sampling runs using the `anesthetic` library.
    
    This class wraps `anesthetic.NestedSamples` to provide a consistent interface
    for accessing posterior samples, weights, and plotting within the `pmrf` framework.
    """
    from anesthetic import NestedSamples

    @property
    def nested_samples(self) -> NestedSamples:
        """
        The underlying anesthetic NestedSamples object.

        Returns
        -------
        anesthetic.NestedSamples
            The object containing the raw nested sampling chains and statistics.
        """
        return self.solver_results
    
    @property
    def sample_param_names(self) -> list[str]:
        """
        The names of the parameters included in the samples.

        This filters out internal columns like 'logL', 'logL_birth', and 'nlive'.

        Returns
        -------
        list of str
            The list of clean parameter names.
        """
        columns = self.nested_samples.columns
        param_names = [columns[i][0] for i in range(len(columns))]
        param_names = [name for name in param_names if name not in {'logL', 'logL_birth', 'nlive'}]
        return param_names    

    def plot_params(self, param_names=None, prior=False, *args, **kwargs):
        """
        Generate a 2D marginal plot (triangle plot) of the parameters.

        Parameters
        ----------
        param_names : list of str, optional
            The list of parameters to plot. Defaults to all model parameters.
        prior : bool, optional, default=False
            If True, overlays the prior distribution on the plot in grey.
        *args
            Additional arguments passed to `make_2d_axes`.
        **kwargs
            Additional keyword arguments passed to `plot_2d`.

        Returns
        -------
        matplotlib.axes.Axes or array of Axes
            The axes grid containing the plots.
        """
        from anesthetic import make_2d_axes
        import matplotlib.pyplot as plt

        param_names = param_names or self.sample_param_names
        fig, axes = make_2d_axes(param_names, *args, **kwargs)
        if prior:
            self.nested_samples.prior().plot_2d(axes, color='grey', alpha=0.5)
        return self.nested_samples.plot_2d(axes)    
    
    def prior_samples(self, equal_weights=False) -> jnp.ndarray:
        """
        Retrieve samples drawn from the prior distribution.

        Parameters
        ----------
        equal_weights : bool, optional, default=False
            If True, returns unweighted (resampled) prior points.
            If False, returns all live points representing the prior.

        Returns
        -------
        jnp.ndarray
            Array of prior samples.
        """
        if equal_weights:
            nested_samples = self.nested_samples.prior_points()
        else:
            nested_samples = self.nested_samples.prior()
        samples = nested_samples.loc[:, self.sample_param_names].to_numpy()
        return jnp.array(samples)
    
    def posterior_samples(self, equal_weights=False) -> jnp.ndarray:
        """
        Retrieve samples drawn from the posterior distribution.

        Parameters
        ----------
        equal_weights : bool, optional, default=False
            If True, returns unweighted (resampled) posterior points (e.g. for histograms).
            If False, returns the raw nested samples (usually required for weighted calculations).

        Returns
        -------
        jnp.ndarray
            Array of posterior samples.
        """
        if equal_weights:
            nested_samples = self.nested_samples.posterior_points()
        else:
            nested_samples = self.nested_samples
        prior_samples = nested_samples.loc[:, self.sample_param_names].to_numpy()
        return jnp.array(prior_samples)

    def weighted_samples(self) -> jnp.ndarray:
        """
        Retrieve the raw samples used for weighted calculations.

        Returns
        -------
        jnp.ndarray
            Array of raw sample parameters.
        """
        return jnp.array(self.nested_samples.loc[:, self.sample_param_names].to_numpy())
    
    def weights(self) -> jnp.ndarray:
        """
        Retrieve the weights corresponding to the posterior samples.

        Returns
        -------
        jnp.ndarray
            Array of weights.
        """
        return jnp.array(self.nested_samples.get_weights())
    
    def prior_weights(self) -> jnp.ndarray:
        """
        Retrieve the weights corresponding to the prior samples.

        Returns
        -------
        jnp.ndarray
            Array of prior weights.
        """
        return jnp.array(self.nested_samples.prior().get_weights())
    
    def encode_solver_results(self, group: h5py.Group):
        """
        Serialize the nested samples to an HDF5 group using CSV format.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group to write the data to.
        """
        samples = self.solver_results
        group['samples'] = samples.to_csv()
        
    @classmethod
    def decode_solver_results(cls, group: h5py.Group) -> Any:
        """
        Deserialize the nested samples from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group to read the data from.

        Returns
        -------
        anesthetic.NestedSamples
            The reconstructed nested samples object.
        """
        from anesthetic import NestedSamples, read_csv
        import pandas as pd
        
        csv_str = group['samples'][()]
        csv_str = csv_str.decode('utf-8') if isinstance(csv_str, bytes) else csv_str
        samples = NestedSamples(read_csv(io.StringIO(csv_str)))
        return samples