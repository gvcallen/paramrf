from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
import skrf as rf

try: 
    import plotly.graph_objs as go
    plotly_available = True
    from plotly.subplots import make_subplots
except ImportError:
    plotly_available = False

try:
    from anesthetic import make_2d_axes, NestedSamples
    anesthetic_available = True
except ImportError:
    anesthetic_available = False

from pmrf.statistics.parameters import ParameterSet
from pmrf.statistics.likelihood import Likelihood
from pmrf.fitting.target import Target
from pmrf.modeling.system import NetworkSystem
from pmrf.core.networks import get_unique_networks, update_networks_mapped
from pmrf.misc.math import dB20
from pmrf.rf.passive import available_gain
from pmrf.plotting import GridFigure

logger = logging.getLogger(__name__)

class Plotter():
    def __init__(self, targets: list[Target], system: NetworkSystem, output_path = None, plot_frequency: rf.Frequency = None, fig_size = (16, 10), fig_dpi = 100, framework = 'matplotlib', nested_samples = None, params: ParameterSet = None, likelihood_object = None, bayesian = False, no_output = False):
        if not output_path is None:
            Path(output_path).mkdir(exist_ok=True, parents=True)
        
        self.frequency = plot_frequency
        if self.frequency is None and not targets is None and len(targets) > 0:
            self.frequency = targets[0].model.frequency
        
        self._targets_original = targets.copy()
        self._targets_active = self._targets_original.copy()  # TODO should we the actual targets here?
        self._system = system
        self.elements = get_unique_networks([target.model for target in self._targets_active], ignore_composite=True, ignore_non_computabe=True)
        
        self.output_path = output_path
        self.fig = None
        self.fig_size = fig_size
        self.fig_dpi = fig_dpi
        self.framework = framework
        self.interactive = False
        self.save = True
        self.is_bayesian = bayesian
        
        self._params_original = params.copy() if params is not None else None
        self._params_active = params.copy() if params is not None else None
        self._nested_samples: NestedSamples = nested_samples
        self._likelihood_object: Likelihood = likelihood_object

        self.no_output = no_output
    
    @property
    def params(self):
        return self._params_active
    
    @params.setter
    def params(self, new: ParameterSet):
        self._params_active = new.copy()
        self._params_original = new.copy()
        
    @property
    def nested_samples(self):
        return self._nested_samples
    
    @property
    def targets(self):
        return self._targets_active
    
    @targets.setter
    def targets(self, new):
        self._targets_active = new

    def set_targets(self, target_names: list[str] = None):
        if target_names is None:
            self._targets_active = self._targets_original.copy()
        else:
            self._targets_active = [target for target in self._targets_original if target.name in target_names]

    def plot_params(self, param_names=None, title='params', label='posterior', priors=False, fig_size=None, kind='kde', bins=None, fig=None, ax=None):
        fig_size = fig_size or self.fig_size
        params, nested_samples = self._params_active, self._nested_samples
        
        kwargs = {
            'kind': kind,
        }
        
        if not bins is None:
            kwargs['bins'] = bins
        
        if not anesthetic_available:
            raise Exception('Anesthetic must be installed to plot corner plots')
        
        if params is None or nested_samples is None:
            raise Exception('Params and nested samples must be passed when initializing a plotter in order to plotting the corner plot')
        
        params = param_names or params.names_free
        
        logger.info('Creating corner plot axes')
        # dim = max(8, len(params))

        if ax is None:
            fig, ax = make_2d_axes(params, figsize=fig_size)

        if fig is None:
            fig = ax.figure

        for i in range(ax.shape[0]):  # Loop over rows
            for j in range(ax.shape[1]):  # Loop over columns
                axi = ax.iloc[i, j]
                axi.set_ylabel(axi.get_ylabel(), rotation='horizontal')

        logger.info('Plotting data')
        
        if priors:
            prior_samples = nested_samples.prior()
            prior_samples.plot_2d(ax, label='pdf', **kwargs)
        
        nested_samples.plot_2d(ax, label=label, **kwargs)
        
        if priors:
            ax.iloc[-1, 0].legend(bbox_to_anchor=(len(ax)/2, len(ax)), loc='lower center', ncol=2)
        
        if self.output_path:
            logger.info('Rendering corner plot')
            fig.savefig(f'{self.output_path}/{title}.png')
            logger.info('Corner plot saved')

        return fig, ax
            
    def plot_available_gain(self, title='Gav', model=True, contours=False, lines=False, R_source=None, source_port=0, fig_size=None):
        fig_size = fig_size or self.fig_size
        
        if contours:
            logger.info('Plotting available gain contours...')
        else:
            logger.info('Plotting available gains...')
        
        layout = [[{"colspan": 2}, None]]
        num_targets = len(self._targets_active)        
        
        if contours or lines:
            samples, weights = self.samples, self.weights
        
        row, col = self._start_figure(num_targets, layout)
        for target in self._targets_active:
            model_title = f"{target.name}"
            
            if hasattr(target.model, 'available_gain'):
                func = lambda _, theta: self.network_from_target(target, theta, update_noise=True).available_gain()
                y = target.model.available_gain()
            else:
                if R_source is None:
                    source = rf.Circuit.Port(frequency=target.model.frequency, name='source', z0=target.model.z0[:,0])
                else:
                    media = rf.media.DefinedGammaZ0(frequency=target.model.frequency, z0_port=target.model.z0[:,0])
                    source = media.resistor(R_source) ** media.short()
                
                func = lambda _, theta: available_gain(source, self.network_from_target(target, theta, update_noise=True), source_port=source_port)
                y = available_gain(source, target.model, source_port=source_port)
                # y = func(None, None)
            
            x = self.frequency.f_scaled
            
            if contours:
                self.fig.plot_contours((row, col), x, func, samples, weights)
            if lines:
                self.fig.plot_lines((row, col), x, func, samples, weights)
            
            if model:
                self.fig.plot((row, col), x, [y], title=model_title, ylabel='Gav', xlabel=f'Frequency ({self.frequency.unit})', colors=['black'], linestyles=['-'])
            row, col = self.fig.next()
                
        self._end_figure(title)  

    def plot_S(self, name='s_params', measured=True, current=True, contours=False, lines=False, mag=True, real_imag=True, port_tuples=None, title=None, fig_size=None):
        model_color, measured_color = ('red', 'black') if not contours else ('blue', 'black')
        model_linestyle, measured_linestyle = ('-', '--')
        
        if contours:
            logger.info('Plotting S-parameter contours...')
        else:
            logger.info('Plotting S-parameters...')
        
        if mag and real_imag:
            port_layout = [[{"colspan": 2}, None], [{}, {}]]
        elif mag:
            port_layout = [[{"colspan": 2}, None]]
        else:
            port_layout = [[{}, {}]]
            
        if port_tuples:
            num_port_plots = len(port_tuples) * len(self._targets_active)
        else:
            num_port_plots = np.sum([target.number_of_ports**2 for target in self._targets_active])
        
        if contours or lines:
            samples, weights = self.samples, self.weights
            
        colors, linestyles = [], []
        if current:
            colors.append(model_color)
            linestyles.append(model_linestyle)
        if measured:
            colors.append(measured_color)
            linestyles.append(measured_linestyle)
            
        row, col = self._start_figure(num_port_plots, port_layout, fig_size=fig_size)
        for target in self._targets_active:
            model_port_tuples = port_tuples or target.model.port_tuples

            for m, n in model_port_tuples:
                if self.is_bayesian:
                    value = f'likelihood = {target.likelihood():.2f}'
                else:
                    value = f'cost = {target.cost():.2f}'
                title = title or f"{target.name}, S{m+1}{n+1} ({value})"
                x = self.frequency.f_scaled
                xlabel = f'Frequency ({self.frequency.unit})'
                                
                # Plot magnitude
                if mag:
                    func_dB = lambda _, theta: self.network_from_target(target, theta, update_noise=True).s_db[:, m, n]
                    if contours:
                        self.fig.plot_contours((row, col), x, func_dB, samples, weights)
                    if lines:
                        self.fig.plot_lines((row, col), x, func_dB, samples, weights)
                    if current or measured:
                        y_vals = []
                        labels = []
                        if current:
                            labels.append('Model')
                            y_vals.append(target.model.s_db[:, m, n])
                        if measured:
                            labels.append('Measured')
                            y_vals.append(target.measured.s_db[:, m, n])
                        self.fig.plot((row, col), x, y_vals, labels=labels, title=title, ylabel='Mag. (dB)', xlabel=xlabel, colors=colors, linestyles=linestyles)
                        
                # Plot real and imaginary
                if real_imag:
                    func_real = lambda _, theta: np.real(self.network_from_target(target, theta, update_noise=True).s[:, m, n])
                    func_imag = lambda _, theta: np.imag(self.network_from_target(target, theta, update_noise=True).s[:, m, n])
                    
                    if mag:
                        row = row + 1
                        title = None
                    if contours:
                        self.fig.plot_contours((row, col), x, func_real, samples, weights)
                        self.fig.plot_contours((row, col+1), x, func_imag, samples, weights)
                    if lines:
                        self.fig.plot_lines((row, col), x, func_real, samples, weights)
                        self.fig.plot_lines((row, col+1), x, func_imag, samples, weights)
                    if current or measured:
                        y_vals_real, y_vals_imag = [], []
                        labels = []
                        if current:
                            y_vals_real.append(target.model.s_re[:, m, n])
                            y_vals_imag.append(target.model.s_im[:, m, n])
                            labels.append('Model')
                        if measured:
                            y_vals_real.append(target.measured.s_re[:, m, n])
                            y_vals_imag.append(target.measured.s_im[:, m, n])
                            labels.append('Measured')
                        self.fig.plot((row, col), x, y_vals_real, labels=labels, title=title, ylabel='Real', xlabel=xlabel, colors=colors, linestyles=linestyles)
                        self.fig.plot((row, col+1), x, y_vals_imag, labels=labels, title=title, ylabel='Imag', xlabel=xlabel, colors=colors, linestyles=linestyles)

                row, col = self.fig.next()
                
                self.fig.legend(loc='lower right')
                
        self._end_figure(name)
        self.reset_params()
    
    def update_params(self, params: np.ndarray | dict, update_networks=True, update_network_likelihoods=True, update_noise=False, scaler=None):
        if params is None:
            return self._system.update_networks()
        elif isinstance(params, dict):
            raise Exception('Updating parameters directly from dict not yet supported')
        
        num_model_params = len(self._system.params.names_free)
        num_likelihood_params = len(params) - num_model_params
        params_networks = params[0:num_model_params]
        if not self.is_bayesian:
            update_network_likelihoods = False
            params_likelihood = None
        else:
            params_likelihood = params[-num_likelihood_params:]

        if update_networks:
            self._system.update_params(params_networks, scaler=scaler)
            for target in self._targets_active:
                target.update_params(params_likelihood, update_noise=update_noise, update_likelihoods=update_network_likelihoods)
    
    def reset_params(self, update_noise=False):
        if self._params_original is not None:
            self._params_active = self._params_original.copy()
        
        self.update_params(None, update_noise=update_noise)

    def network_from_target(self, target: Target, theta = None, update_noise=False):
        self.update_params(theta, update_noise=update_noise)
        return target.model
    
    def _start_figure(self, num_items, item_layout, fig_size=None):
        fig_size = fig_size or self.fig_size
        
        if not self.interactive or (self.interactive and not self.fig):
            self.fig = GridFigure(num_items=num_items, item_layout=item_layout, framework=self.framework, interactive=self.interactive, fig_size=fig_size, fig_dpi=self.fig_dpi)
            
        return self.fig.start()
    
    def _end_figure(self, title):
        self.fig.tight_layout()
        
        if self.save:
            self.fig.save(self.output_path, title)            

    @property
    def nested_samples(self):
        return self._nested_samples
    
    @property
    def samples(self):
        if self._nested_samples is None or self._params_active is None:
            raise Exception('Nested samples and parameters must be passed to plot contours and other Bayesian plots')
        
        params = self._params_active.names_free + self._likelihood_object.param_names()
        nested_samples = self._nested_samples
        
        samples = nested_samples.loc[:, params].to_numpy()
        return samples
    
    @property
    def weights(self):
        if self._nested_samples is None:
            raise Exception('Nested samples and parameters must be passed to plot contours and other Bayesian plots')
        
        nested_samples = self._nested_samples
        weights = nested_samples.get_weights()
        return weights    
    