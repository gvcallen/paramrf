import numpy as np
import skrf as rf

from pmrf.statistics.features import Feature, extract_features
from pmrf.statistics.modifiers import ModifierChain
from pmrf.statistics.likelihood import GaussianLikelihood

from pmrf.core.networks import ParametricNetwork, CompositeNetwork
from pmrf.core.core import add_noise

class Target:
    """
    A fitting "Target" is a higher-level representation of the model and its measured data. The model network is encapsulated into this class,
    which also contains the reference measured network, which features to extract from the model, and cost function information.
    """
    def __init__(self, model: ParametricNetwork, measured: rf.Network = None, use_measured = False, fixed = False,
                 features: list[Feature] = [Feature('complex')], cost_chain: ModifierChain = ModifierChain(['L2-M', 'dB']), likelihood_object = GaussianLikelihood()):
        self._model = model
        self._model.fixed = fixed
        self._measured = measured
        
        self.use_measured = use_measured
        self.features_list = features
        self.cost_chain = cost_chain
        self.likelihood_object = likelihood_object
        
    @property
    def port_tuples(self):
        target_port_tuples = self._model.port_tuples
        measured_port_tuples = self._measured.port_tuples
        port_tuples = list(set(target_port_tuples).intersection(measured_port_tuples))
        return sorted(port_tuples)

    @property
    def name(self):
        return self._model.name
    
    @property
    def fixed(self):
        return self._model.fixed
    
    @fixed.setter
    def fixed(self, value):
        self._model.fixed = value

    @property
    def number_of_ports(self):
        return min(self._model.number_of_ports, self._measured.number_of_ports)

    @property
    def nports(self):
        return self.number_of_ports
    
    @property
    def measured(self):
        return self._measured
    
    @property
    def model(self) -> ParametricNetwork:
        if self.use_measured:
            return self._measured
        else:
            return self._model
            
    @property
    def frequency(self) -> rf.Frequency:
        return self._model.frequency
    
    def features(self, measured = False, ignore_imag=False) -> np.ndarray:
        # Extracts "features" from the model (S11 real, S21 complex etc.) and returns as a F x D array,
        # where F is the number of frequencies, and D is the number of features
        if measured:
            network = self.measured
        else:
            network = self.model
        y = extract_features(network, self.features_list, ignore_imag=ignore_imag)
        return y

    def cost(self) -> float:
        y_meas = self.features(measured=True)
        y_model = self.features()
        y_diff = y_meas - y_model
        return self.cost_chain(y_diff)
    
    def likelihood(self) -> float:
        y_meas = self.features(measured=True)
        y_model = self.features()
        return self.likelihood_object(y_meas, y_model)
    
    def update_params(self, params: np.ndarray | dict, update_likelihoods=False, update_noise=False):
        if self.fixed:
            return
        
        if isinstance(params, dict):
            if update_likelihoods:
                likelihood_params = self.likelihood_object.params()
                likelihood_params.update(params)
                self.likelihood_object.update_params(list(likelihood_params.values()))
            if update_noise:
                if 'sigma' in params:                
                    self.model.sigma_tau = params['sigma']
                    self.model.sigma_gamma = params['sigma']
                else:
                    self.model.sigma_tau = params['sigma_tau']
                    self.model.sigma_gamma = params['sigma_gamma']
            else:
                self.model.sigma_gamma = 0.0
                self.model.sigma_tau = 0.0
        else:
            if update_likelihoods:
                self.likelihood_object.update_params(params)

            if update_noise:
                if not update_likelihoods:
                    raise Exception('Must update network likelihoods when updating noise')
                sigma = self.likelihood_object.params()['sigma']
                self.model.sigma_gamma = sigma
                self.model.sigma_tau = sigma