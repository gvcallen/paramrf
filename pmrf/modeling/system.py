from pathlib import Path
import logging
import os

import numpy as np
import skrf as rf

from pmrf.modeling.networks import ParametricNetwork, CompositeNetwork, update_networks_mapped, get_unique_networks
from pmrf.statistics import ParameterSet
from pmrf.misc.math import round_sig

VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")
# Add a method to the logger for convenience
def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)
logging.Logger.verbose = verbose

logger = logging.getLogger(__name__)


class CircuitSystemSettings:
    """
    Settings for a circuit system. This currently only includes the frequency range, but will for for configuring sampling settings for simulations etc.
    The recommended way to initialize all these settings is simply by passing them as kwargs directly to the "CircuitSystem" class or sub-class.
    """
    def __init__(self, frequency=None, **kwargs):       
        self.param_path = None                                                          # The input parameters. Must be a string pointing to the input .csv file. Can be None, in which case a ParameterSet must be passed directly to the system.
        
        # Parameter settings
        self.param_deviation = 0.1                                                      # Parameter bounds deviation from nominal values. Only used when parameter path or set is not passed.
        self.param_sig_fig = 4                                                          # Significant figures for initial parameter values. Only used when parameter path or set is not passed.
        self.param_infix = '_'                                                          # Infix used to map parameters from global names to model-local names.
        
        # Network settings
        self.frequency = frequency                                                      # The frequency shared by all networks


class CircuitSystem:
    """
    A circuit system is a collection of dependent parametric networks. It encapsulates all the model parameters and allows updating and sampling the parameters.
    Used by the CircuitFitter. Also useful for simulating models.
    """
    def __init__(self, networks: list[ParametricNetwork] = None, settings: CircuitSystemSettings = None, param_set: ParameterSet = None, **kwargs):
        """The initializer for a CircuitSystem.

        Args:
            networks (list[ParametricNetwork], optional): Specifies the networks to use. Defaults to None, which is useful for derived classes, where they should initialize their models in _init_networks().
            settings (CircuitSystemSettings, optional): A setting struct to initialize settings from. Generally key-word arguments are passed instead. Defaults to None.
            param_set (ParameterSet, optional): A ParameterSet object to load parameters from. Useful for defining parameters in code as opposed to loading them from a .csv. Defaults to None.
            **kwargs: Key-word arguments. This is the main way to configure the class. Possible arguments are all members of the CircuitSystemSettings classes.
        """
        self._settings = settings or CircuitSystemSettings(**kwargs)
        self._networks = networks or []
        self._params_original: ParameterSet = None
        self._params_active: ParameterSet = None
        self._frequency = self._settings.frequency
        
        self._init_networks()
        self._subnetworks = get_unique_networks(self._networks, ignore_composite=True, ignore_non_computabe=True)
        self._init_params(param_set=param_set)
        self._init_frequencies()     
        
    def _init_networks(self):
        pass        
        
    def _init_params(self, param_set=None):
        logger.verbose("Initializing parameters")
        
        # Create params if necessary
        if self._settings.param_path is None and param_set is None:
            data = []
            for network in self._networks:
                param_ntwks = network.subnetworks_recursive(ignore_composite=True, ignore_non_computable=True)
                for ntwk in param_ntwks:
                    for name, value in ntwk.params.items():
                        name = ntwk.name + self._settings.param_infix + name
                        scale = 1.0
                        fixed = False
                        min, max = value * (1.0 - self._settings.param_deviation), value * (1.0 + self._settings.param_deviation)
                        min, max = round_sig(min, self._settings.param_sig_fig), round_sig(max, self._settings.param_sig_fig)

                        if self._settings.param_type == 'uniform':
                            data.append([name, value, scale, fixed, min, max])
                        else:
                            raise Exception('Only uniform priors accepted as defaults')
            param_set = ParameterSet(data=data, columns=['name', 'value', 'scale', 'fixed', 'minimum', 'maximum'])           

        self.load_params(file=self._settings.param_path, param_set=param_set)
        
    def _init_frequencies(self):
        for network in self._networks:
            network.notifying = False
        for network in self._networks:
            network.interpolate_self(self._settings.frequency)
        for network in self._networks:
            network.notifying = True                
        
    @property
    def networks(self) -> list[rf.Network]:
        return self._networks.copy()
    
    @property
    def subnetworks(self) -> list[rf.Network]:
        return self._subnetworks.copy()
    
    @property
    def params(self) -> ParameterSet:
        return self._params_active
    
    @property
    def params_original(self) -> ParameterSet:
        return self._params_original

    @property
    def param_names(self) -> list[str]:
        return self.params.index.to_list()
    
    @property
    def num_free_params(self):
        return len(self.params.names_free)        
    
    @property
    def frequency(self) -> rf.Frequency:
        return self._frequency
    
    def update_networks(self):
        update_networks_mapped(self._subnetworks, self.params.evaluate())

    def update_params(self, params: np.ndarray | dict, scaler = None, hypercube = False):
        if isinstance(params, dict):
            raise Exception('Updating parameters directly from dict not yet supported')

        if scaler:
            params = scaler.inverse_transform(params)
        if hypercube:
            params = np.array([pdf(params[i]) for i, pdf in enumerate(self.params.pdfs())])

        self.params.update_values(params)
        update_networks_mapped(self._subnetworks, self.params.evaluate())
        
    def load_params(self, file=None, param_set=None):
        if not file is None:
            self._params_original = ParameterSet(file=file)
        else:
            self._params_original = param_set.copy()
        
        if len(self._params_original.scale[self._params_original.scale == 0.0]):
            raise Exception("Error: parameter(s) with zero scale")        
        
        self.reset_params()
            
    def save_params(self, file='params.csv'):
        dir_path = os.path.dirname(file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        self.params.flush_cache()

        # We don't save parameter_active directly as it has the fixed flags changed, as well as potentially new parameters/columns.
        # Instead, we just copy its updated values to the original parameters and save that
        parameters_save = self._params_original.copy()
        parameters_save.value = self._params_active.loc[self._params_active.index.isin(parameters_save.index)].value
        parameters_save.write_csv(f'{file}')

        logger.verbose('Parameters saved to file')
        self.params.enable_cache()        
        
    def reset_params(self):
        self._params_active = self._params_original.copy()

        # We set all parameter to be fixed first, and only set them to be free (non-fixed) if at least one model has them free,
        # and they are also free in the original parameters.
        self._params_active.fixed = True
        
        params_not_found = []

        # Enable parameter opt flags based on target flags.
        for network in self.networks:
            subnetworks: list[ParametricNetwork] = get_unique_networks([network], ignore_composite=True, ignore_non_computabe=True)
            for subnetwork in subnetworks:
                for param_name in subnetwork.params_mapped().keys():
                    param_found = False
                    # TODO this is messy and should be cleaned up - the System class (somehow) shouldn't have to deal with derived parameter
                    try:
                        param_value = self._params_original.loc[param_name].value
                        param_found = True
                        try:
                            _ = float(param_value)
                        except:
                            param_name = self._params_original.loc[param_name].value
                    except:
                        pass
                    
                    if param_found == False:
                        params_not_found.append(param_name)

                    network_fixed = network.fixed
                    subnetwork_fixed = subnetwork.fixed
                    try:
                        param_fixed = self._params_original.loc[param_name].fixed
                    except:
                        param_fixed = True
                    fixed = network_fixed or subnetwork_fixed or param_fixed

                    if not fixed:
                        self._params_active.loc[param_name, 'fixed'] = False
                        
        if len(params_not_found) != 0:
            logger.warning(f'\n\nWARNING: The following parameters were not found and will be fixed: {sorted(list(set(params_not_found)))}\n\n')

        self.update_networks()
        
    def save_touchstone(self, path, networks=True, subnetworks=False, export_frequency=None):
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        logger.verbose('Saving touchstone data to file...')

        export_frequency = export_frequency or self._settings.frequency
        
        if networks and subnetworks:
            networks_path = f'{path}/networks'
            subnetworks_path = f'{path}/subnetworks'
        else:
            networks_path = subnetworks_path = path

        if networks:
            for network in self._networks:
                Path(networks_path).mkdir(exist_ok=True, parents=True)
                
                if not export_frequency is None:
                    network = network.interpolate(export_frequency)
                
                network.write_touchstone(f'{networks_path}/{network.name}')

        if subnetworks:
            for subnetwork in self._subnetworks:
                Path(subnetworks_path).mkdir(exist_ok=True, parents=True)

                if not export_frequency is None:
                    subnetwork = subnetwork.interpolate(export_frequency)

                subnetwork.write_touchstone(f'{subnetworks_path}/{subnetwork.name}')