from abc import abstractmethod
from copy import copy, deepcopy
from typing import Self

import numpy as np
import skrf as rf
from skrf.media import DefinedGammaZ0
from skrf.media import Media

from paramrf.misc.structures import ObservableDict
from paramrf.misc.inspection import get_properties_and_attributes

def add_noise(network: rf.Network, sigma_gamma=0.1, sigma_tau=None):
    num_freq = network.frequency.npoints
    
    if sigma_tau is None:
        sigma_tau = sigma_gamma
    
    for m, n in network.port_tuples:
        if m == n:
            sigma = sigma_gamma
        else:
            sigma = sigma_tau
        noise = sigma * np.random.randn(num_freq) + 1j*sigma * np.random.randn(num_freq)                

        network.s[:, m, n] += noise

class ComputableNetwork(rf.Network):
    """
    A computable network is one which has internal properties that can be computed using some procedure (concatenating other networks, parametrically etc.).
    It therefore has a "compute" method which must be over-ridden.
    """
    def __init__(self, frequency = rf.Frequency(1, 2, 2, 'MHz'), nports: int = 1, z0_port = 50.0, sigma_tau = 0.0, sigma_gamma=0.0, media = None, *args, **kwargs):
        self.z0_port = z0_port
        self._initialized = False
        self._fixed = False
        self._frozen = False
        self._version = 0
        self._sigma_gamma = sigma_gamma
        self._sigma_tau = sigma_tau
        
        s = np.zeros((frequency.npoints, nports, nports))
        z0 = z0_port
        super().__init__(frequency=frequency, s=s, z0=z0, *args, **kwargs)

        self._media = media or DefinedGammaZ0(self.frequency, z0_port=self.z0_port)
        self._initialized = True

    def interpolate_self(self, frequency: rf.Frequency, **kwargs) -> None:
        self._media.frequency = frequency

        if self._fixed:
            return super().interpolate_self(frequency, **kwargs)
        else:        
            nports = self.nports        
            s = np.zeros((frequency.npoints, nports, nports))
            z0 = np.ones((frequency.npoints, nports))
            z0 = z0 * self.z0[0,:]
        
            self.frequency, self.s, self.z0 = frequency, s, z0
            self.update()
            
    def interpolate(self, frequency: rf.Frequency, **kwargs) -> Self:
        if self._fixed:
            return super().interpolate(frequency, **kwargs)
        else:
            ntwk = self.copy()
            ntwk.interpolate_self(frequency, **kwargs)
            return ntwk
    
    @property
    def uncomputable(self) -> rf.Network:
        return rf.Network(name=f'{self.name}_fixed', frequency=self.frequency, s=self.s, z0=self.z0_port)
    
    def __pow__(self, other: rf.Network) -> rf.Network:
        return self.uncomputable ** other
        
    def update(self):
        if self._initialized and not self._fixed and not self._frozen:
            # Base compute
            self.compute()
            
            # Incremenet version number
            self._version += 1
            
    def __setattr__(self, attr_name, attr_value):
        super().__setattr__(attr_name, attr_value)
        
        if hasattr(self, '_initialized') and self._initialized:
            if (self._sigma_gamma != 0.0 or self._sigma_tau != 0.0) and attr_name in self.PRIMARY_PROPERTIES:
                add_noise(self, self._sigma_gamma, self._sigma_tau)
            
    @property
    def sigma(self):
        if self._sigma_gamma != self._sigma_tau:
            raise Exception('Error: sigma getter can only be used when sigma gamma and sigma tau are equal')
            
        return self._sigma_gamma
    
    @sigma.setter
    def sigma(self, value):
        if self._sigma_gamma != value or self._sigma_tau != value:
            self._sigma_gamma = value
            self._sigma_tau = value
            self.compute()            
            
    @property
    def sigma_gamma(self):
        return self._sigma_gamma
    
    @sigma_gamma.setter
    def sigma_gamma(self, value):
        if self._sigma_gamma != value:
            self._sigma_gamma = value
            self.compute()

    @property
    def sigma_tau(self):
        return self._sigma_tau
    
    @sigma_tau.setter
    def sigma_tau(self, value):
        if self._sigma_tau != value:
            self._sigma_tau = value
            self.compute()

    @property
    def version(self):
        return self._version
            
    @property
    def media(self) -> Media:
        return self._media

    @property
    def fixed(self):
        return self._fixed
    
    @fixed.setter
    def fixed(self, fixed):
        self._fixed = fixed
        if not fixed:
            self.update()

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        if self._frozen == True:
            self._frozen = False
            self.update()         

    @abstractmethod
    def compute(self):
        pass


class ObservableNetwork(ComputableNetwork):
    """
    An observable network is a simply a network that implements the observer pattern. In other words, it allows external classes to register as an "observer",
    and then whenever the observer network changes internally, it notifies all registered observers.
    """
    def __init__(self, *args, **kwargs):
        self._observers: list[ObservableNetwork] = []
        self._notifying = True
        super().__init__(*args, **kwargs)
    
    def add_observer(self, component):
        self._observers.append(component)
    
    def remove_observer(self, component):
        self._observers.remove(component)

    def notify_observers(self):
        for observer in self._observers:
            observer.update()

    @property
    def notifying(self):
        return self._notifying
    
    @notifying.setter
    def notifying(self, value):
        if value != self._notifying:
            self._notifying = value
            if value == True:
                self.notify_observers()

    @property
    def observers(self):
        return copy(self._observers)

    def update(self):
        super().update()
        if self._notifying:
            self.notify_observers()
    
    
class ParametricNetwork(ObservableNetwork):
    """
    A parametric network is a network that is computed based on a set of parameters.
    This class is an abstract base class, and only provides convenient functionality to make it easy to define sub-classes.
    
    To use, simply override the __init__() and compute() methods. In the sub-classes __init__(), super().__init__() should be called
    with a dictionary of parameters and the number of ports of the network, and in the sub-classes compute(), one of the network matrices
    (e.g. the S-matrix in self.s) should be filled appropriately. After super().__init__() is called, all parameters can be access via self.params[key] or self.key.
    Note that to avoid conflicts with the base class, it is recommend to start all parameters with capital letters.
    """
    def __init__(self, params: dict = None, **kwargs) -> None:        
        params = params or {}
        self._params = ObservableDict(owner=self)
        self._params.add_update_callback(lambda _: self.update())
        self._params.add_set_callback(lambda _1, _2, _3: self.update())
        
        super().__init__(params=params, **kwargs)
        
        # This not worth it in case of name conflicts with parent class
        props = get_properties_and_attributes(self.__class__)
        for key in params.keys():
            if key in props:
                raise Exception(f'Error: cannot create ParametricNetwork with key {key} as it already exists in the parent class')
        self.update()

    def __setattr__(self, attr_name, attr_value):
        if '_params' in self.__dict__ and attr_name in self._params:
            self._params[attr_name] = attr_value
        else:
            super().__setattr__(attr_name, attr_value)

    def __getattr__(self, attr_name):
        if '_params' in self.__dict__ and attr_name in self._params:
            return self._params[attr_name]
        else:
            return super().__getattr__(attr_name)
        
    def copy(self, *, shallow_copy = False):
        if shallow_copy:
            return copy(self)
        return deepcopy(self)
        
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        setattr(result, '_initialized', False)
        setattr(result, '_params', ObservableDict(owner=result))
        setattr(result, '_observers', [])

        for key, value in self.__dict__.items():
            if key in ['_params', '_observers', '_initialized']:
                continue
            else:
                setattr(result, key, deepcopy(value, memo))
        
        setattr(result, '_initialized', True)
        
        result._params.add_update_callback(lambda _: result.update())
        result._params.add_set_callback(lambda _1, _2, _3: result.update())        
        result._params.update(self._params)
        return result

    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, new_params: dict):
        self._params.update(new_params)        

    def params_mapped(self, infix='_') -> dict[str, float]:
        params = self.params
        prefix = self.name + infix
        params_global = {prefix + k: v for k, v in params.items()}
        return params_global          
        
    def update_mapped(self, params_global: dict | np.ndarray, infix='_'):
        if isinstance(params_global, dict):
            prefix = self.name + infix
            
            # params_local = {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix) and k[len(prefix):] in self.model.params}
            # self.model.params.update(params_local)
            
            params_local = dict(zip(self.params.keys(), self.params.values()))
            for key in self.params.keys():
                try:
                    params_local[key] = params_global[prefix+key]
                except:
                    pass

            self.params.update(params_local)
        else:
            self.params.update(dict(zip(self.params.keys(), params_global)))      


class CompositeNetwork(ParametricNetwork):
    """
    A composite network is a network that is built up using other networks, as well as potential parameters. It is therefore different
    to a parametric network in that its update is not only triggered by a change in its parameters, but also by any change in any sub-network.
    Composite networks can also contain other composite networks, but these have to be "observable" networks.

    Also note that a composite network itself is a parametric network. The parameters of the network have prefixes that are the names of each
    sub-network, and suffixes that are the parameters in each sub-network.
    
    Regarding implementation details, this class requires, as mentioned, that it is built up from observable networks, such that it can register itself
    as an observer and update its internals when any network it depends on is updated. It itself is also an observer, such that a graph
    of whatever complexity of dependencies can be built up (composite networks of composite networks etc.) and they all stay up-to-date
    when the parameter of an parametric network down the dependencie
    """
    def __init__(self, subnetworks: dict[str, ObservableNetwork], mapped_params=False, init_subnetwork_frequencies=False, **kwargs) -> None:    
        found_names = []
        for network in subnetworks.values():
            if network.name in found_names:
                raise Exception('Sub-networks in a CompositeNetwork must all have unique names')
            found_names.append(network.name)
            
        self._mapped_params = mapped_params
        self.propagate_subnetworks = True
        if init_subnetwork_frequencies:
            frequency = kwargs['frequency']
        else:
            frequency = None
        self._init_subnetworks(subnetworks, subnetwork_frequency=frequency)

        if mapped_params:
            params = self._init_mapped_params()
            super().__init__(params, **kwargs)
            self._params.remove_callbacks()
            self._params.add_set_callback(lambda key, value, _: self._set_subnetwork_param(key, value))
            self._params.add_update_callback(lambda _: self._set_subnetwork_params())
        else:
            super().__init__(**kwargs)
        
        super().update()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Dont copy params, subnetworks and observers
        setattr(result, '_initialized', False)
        
        setattr(result, '_subnetworks', ObservableDict(owner=result))
        setattr(result, '_params', ObservableDict(owner=result))
        setattr(result, '_observers', [])
        for key, value in self.__dict__.items():
            if key in ['_subnetworks', '_params', '_observers', '_initialized']:
                continue
            else:
                setattr(result, key, deepcopy(value, memo))

        setattr(result, '_initialized', True)

        # Populate subnetworks with a deepcopy of the current subnetworks
        subnetworks = dict(zip(deepcopy(list(self._subnetworks.keys())), deepcopy(list(self._subnetworks.values()))))
        result._init_subnetworks(subnetworks)
        
        # Add the parameter callbacks (including those that ParametricNetwork adds)
        result._params.add_set_callback(lambda key, value, _: result._set_subnetwork_param(key, value))
        result._params.add_update_callback(lambda _: result._set_subnetwork_params())
    
        # Update the parameters with the         
        result._params.update(self._init_mapped_params())

        return result

    @property
    def subnetworks(self) -> list[ObservableNetwork]:
        return list(self._subnetworks.values())
    
    @property
    def notifying(self):
        return ParametricNetwork.notifying

    @notifying.setter
    def notifying(self, value):
        if self.propagate_subnetworks:
            for network in self._subnetworks.values():
                network.notifying = value
            
        return ParametricNetwork.notifying.__set__(self, value)

    def interpolate_self(self, frequency: rf.Frequency, **kwargs) -> None:
        if self.propagate_subnetworks:
            for network in self._subnetworks.values():
                network.interpolate_self(frequency, **kwargs)
        
        super().interpolate_self(frequency, **kwargs)

    def freeze(self):
        if self.propagate_subnetworks:
            for network in self._subnetworks.values():
                if isinstance(network, ComputableNetwork):
                    network.freeze()
        
        super().freeze()

    # The following code provides a massive performance boost. Effectively, if dependent networks of this network are updated one-by-one,
    # then this network will be re-computed an unnecessary amount of times.
    def update(self):
        needs_update = False
        new_versions = [network.version for network in self._subnetworks.values() if isinstance(network, ObservableNetwork)]
        for new_version, prev_version in zip(new_versions, self._subnetwork_versions):
            if new_version != prev_version:
                needs_update = True
                break
        
        self._subnetwork_versions = new_versions

        if needs_update:
            super().update()
            
    def detach(self):
        self.fixed = True
        for network in self.subnetworks:
            network.remove_observer(self)
        
    def subnetwork_by_name(self, name) -> ObservableNetwork:
        for network in self._subnetworks.values():
            if network.name == name:
                return network
        return None

    def get_subnetworks(self, ignore_composite = False, ignore_non_computable = False):
        subnetworks_list = []
        _append_subnetworks(self, subnetworks_list, ignore_composite=ignore_composite, ignore_non_computable=ignore_non_computable)
        return subnetworks_list
    
    def __getattr__(self, name):
        if '_subnetworks' in self.__dict__ and name in self._subnetworks:
            return self._subnetworks[name]
        else:
            return super().__getattr__(name)

    def __setattr__(self, name, value):
        if '_subnetworks' in self.__dict__ and name in self._subnetworks:
            self._subnetworks[name] = value
        else:
            super().__setattr__(name, value)        
   
    def _set_subnetwork_param(self, key, value) -> bool:
        # Set the parameter for the subnetwork, if it is not the same value
        # TODO this will call update_param_from_subnetwork for no reason
        subnetwork, prefix = next(((ntwk, ntwk.name) for ntwk in self._subnetworks.values() if key.startswith(ntwk.name)), None)
        subparam = key[len(prefix)+1:]
        changed = False
        if subnetwork.params[subparam] != value:
            subnetwork.params[subparam] = value
            changed = True
        
        return changed

    def _set_subnetwork_params(self):
        self.freeze()

        # TODO sort params by subnetwork instead of looking up subnetwork for each param
        changed = False
        for key, value in self._params.items():
            if self._set_subnetwork_param(key, value):
                changed = True
        
        self.unfreeze()
        if changed:
            super().update()

    def _init_subnetworks(self, subnetworks: dict[str, ObservableNetwork], subnetwork_frequency=None):
        if subnetwork_frequency:
            for network in subnetworks.values():
                network.interpolate_self(subnetwork_frequency)
        
        self._subnetworks = ObservableDict(owner=self)
        self._subnetworks.update(subnetworks)
        self._subnetworks.add_update_callback(lambda _: self.update())
        self._subnetworks.add_set_callback(lambda _1, _2, _3: self.update())
        self._subnetwork_versions: list[int] = []
        
        props = get_properties_and_attributes(self.__class__)
        for subnetwork in self._subnetworks.values():
            if not isinstance(subnetwork, ObservableNetwork):
                continue

            # Add as observed network
            subnetwork.add_observer(self)
            self._subnetwork_versions.append(subnetwork._version)
            
            # Add name as property
            name = subnetwork.name
            if name in props:
                raise Exception(f'Error: cannot create CompositeNetwork with subnetwork named {name} as this property already exists in the parent class')                 

    def _init_mapped_params(self):
        params = {}

        for subnetwork in self._subnetworks.values():
            if isinstance(subnetwork, ParametricNetwork):
                for param_name, param_value in subnetwork.params.items():
                    param_name_mapped = f'{subnetwork.name}_{param_name}'
                    params[param_name_mapped] = param_value

                def update_param_from_subnetwork(key, value, obs_dict):
                    notify_before = self._params.notify
                    self._params.notify = False
                    self._params[f'{obs_dict.owner.name}_{key}'] = value
                    self._params.notify = notify_before
                
                def update_params_from_subnetwork(obs_dict):
                    notify_before = self._params.notify
                    self._params.notify = False
                    for key in obs_dict.owner._params.keys():
                        value = obs_dict.owner._params[key]
                        update_param_from_subnetwork(key, value, obs_dict)
                    self._params.notify = notify_before
                
                subnetwork._params.add_set_callback(update_param_from_subnetwork)
                subnetwork._params.add_update_callback(update_params_from_subnetwork)               

        return params
            
            
def _append_subnetworks(network: CompositeNetwork, subnetworks: list[rf.Network], ignore_composite = False, ignore_non_computable = False):
    for ntwk in network.subnetworks:
        if isinstance(ntwk, CompositeNetwork):
            if not ignore_composite:
                subnetworks.append(ntwk)
            _append_subnetworks(ntwk, subnetworks, ignore_composite=ignore_composite)
        else:
            if ignore_non_computable:
                if isinstance(ntwk, ComputableNetwork):
                    subnetworks.append(ntwk)
            else:
                subnetworks.append(ntwk)
                
def get_unique_networks(networks: list[rf.Network], ignore_composite=False, ignore_non_computabe=False) -> list[rf.Network]:
    if networks is None:
        return []
    
    subnetworks = []
    for network in networks:
        if isinstance(network, CompositeNetwork):
            subnetworks.extend(network.get_subnetworks(ignore_composite=ignore_composite, ignore_non_computable=ignore_non_computabe))
            if not ignore_composite:
                subnetworks.append(network)
        elif isinstance(network, ComputableNetwork) or not ignore_non_computabe:
            subnetworks.append(network)
            
    seen = {}
    subnetworks = [seen.setdefault(ntwk.name, ntwk) for ntwk in subnetworks if ntwk.name not in seen]
    return subnetworks

def update_networks_mapped(networks: list[ParametricNetwork], params_global: dict, infix='_'):
    for network in networks:
        network.notifying = False
    for network in networks:
        network.update_mapped(params_global, infix=infix)
    for network in networks:
        network.notifying = True