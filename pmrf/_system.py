from abc import abstractmethod, ABC
from pathlib import Path
import logging
import os
from dataclasses import dataclass, field

from pmrf._numpy import numpy as np
import skrf

from pmrf.statistics.parameters import ParameterSet
from pmrf._math import round_sig
from pmrf._model import Model

import equinox as eqx

VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")
# Add a method to the logger for convenience
def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)
logging.Logger.verbose = verbose

logger = logging.getLogger(__name__)


class SystemModel(Model):
    """ A `SystemModel` is a collection of models which itself is a model.

    It provides an easy abstract class to be derived that can model a set of related models that logically form a system.
    For example, one might want to model a high-frequency circuit board with a number of different inputs but with some transmission lines
    in the path shared. In that case, in might make sense to model these inputs as separate models that each include the same underlying
    transmission line model.

    The `SystemModel` overrides some of the default model methods with those more tailored towards shared models,
    making it a useful abstraction for general purposes.
    """
    models: list[Model] = eqx.field(default_factory=lambda: [])
    name: str | None = eqx.field(default=None)
    
    def to_skrf(self, frequency: skrf.Frequency | list[skrf.Frequency], **kwargs) -> skrf.NetworkSet:
        networks = []

        if not isinstance(frequency, list):
            frequency = [frequency] * len(self.models)

        if isinstance(frequency, list):
            for model, model_frequency in zip(self.models, frequency):
                networks.append(model.to_skrf(model_frequency, **kwargs))
        else:
            model_frequency = frequency
            for model in self.models:
                networks.append(model.to_skrf(model_frequency, **kwargs))
        
        return skrf.NetworkSet(ntwk_set=networks, name=self.name)
        
    @property
    def number_of_models(self):
        return len(self.models)
    
    @property
    def n_models(self):
        return self.number_of_models

    @property
    def submodels(self) -> 'SystemModel':
        raise Exception('Not yet implemented')
        # for model in self.models:


    # def save_params(self, file='params.csv'):
            
    #     # We don't save parameter_active directly as it has the fixed flags changed, as well as potentially new parameters/columns.
    #     # Instead, we just copy its updated values to the original parameters and save that
    #     parameters_save = self._params_original.copy()
    #     parameters_save.value = self._params.loc[self._params.index.isin(parameters_save.index)].value
    #     parameters_save.write_csv(f'{file}')

    #     logger.verbose('Parameters saved to file')
        
    # def reset_params(self):
    #     self._params = self._params_original.copy()

    #     # We set all parameter to be fixed first, and only set them to be free (non-fixed) if at least one model has them free,
    #     # and they are also free in the original parameters.
    #     self._params.fixed = True
        
    #     params_not_found = []

    #     # Enable parameter opt flags based on target flags.
    #     for network in self.networks:
    #         subnetworks: list[Model] = get_unique_networks([network], ignore_composite=True, ignore_non_computabe=True)
    #         for subnetwork in subnetworks:
    #             for param_name in subnetwork.params_global(self.settings.param_infix).keys():
    #                 param_found = False
    #                 # TODO this is messy and should be cleaned up - the System class (somehow) shouldn't have to deal with derived parameter
    #                 try:
    #                     param_value = self._params_original.loc[param_name].value
    #                     param_found = True
    #                     try:
    #                         _ = float(param_value)
    #                     except:
    #                         param_name = self._params_original.loc[param_name].value
    #                 except:
    #                     pass
                    
    #                 if param_found == False:
    #                     params_not_found.append(param_name)

    #                 network_fixed = network.fixed
    #                 subnetwork_fixed = subnetwork.fixed
    #                 try:
    #                     param_fixed = self._params_original.loc[param_name].fixed
    #                 except:
    #                     param_fixed = True
    #                 fixed = network_fixed or subnetwork_fixed or param_fixed

    #                 if not fixed:
    #                     self._params.loc[param_name, 'fixed'] = False
                        
    #     if len(params_not_found) != 0 and not self._warning_emitted:
    #         logger.warning(f'WARNING: The following parameters were not found and will be fixed: {sorted(list(set(params_not_found)))}')
    #         self._warning_emitted = True

    #     self.update_networks()

    # def save_touchstone(self, path, frequency, models=True):
    #     dir_path = os.path.dirname(path)
    #     if dir_path:
    #         os.makedirs(dir_path, exist_ok=True)
        
    #     logger.verbose('Saving touchstone data to file...')

    #     if models and subnetworks:
    #         networks_path = f'{path}/networks'
    #         subnetworks_path = f'{path}/subnetworks'
    #     else:
    #         networks_path = subnetworks_path = path

    #     if models:
    #         for network in self.models:
    #             Path(networks_path).mkdir(exist_ok=True, parents=True)
    #             network.to_skrf(frequency).write_touchstone(f'{networks_path}/{network.name}')

    #     if subnetworks:
    #         raise Exception('Saving of subnetworks to file is currently unsupported')
    #         # for subnetwork in self._subnetworks:
    #         #     Path(subnetworks_path).mkdir(exist_ok=True, parents=True)

    #         #     if not frequency is None:
    #         #         subnetwork = subnetwork.interpolate(frequency)

    #         #     subnetwork.write_touchstone(f'{subnetworks_path}/{subnetwork.name}')