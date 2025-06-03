from pathlib import Path
import glob
import shutil
import logging
import sys
import json

import uuid
from datetime import timedelta
import time

import numpy as np
from scipy.optimize import minimize, Bounds, shgo
import skrf as rf

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    mpi_available = True
except:
    rank = 0
    mpi_available = False

try:
    from anesthetic import read_chains
    anesthetic_available = True
except ImportError:
    anesthetic_available = False

try:
    import pypolychord
    polychord_available = True
except ImportError:
    polychord_available = False

from pmrf.misc.structures import frequency_to_dict, dict_to_frequency
from pmrf.misc.other import time_string
from pmrf.misc.math import round_sig

from pmrf.statistics.features import Feature, extract_features
from pmrf.statistics.modifiers import ModifierChain
from pmrf.statistics.likelihood import GaussianLikelihood, CircularComplexGaussianLikelihood, RicianLikelihood
from pmrf.statistics.pdf import UniformPDF

from pmrf.fitting.target import Target

from pmrf.plotting import Plotter
from pmrf.core import ParametricNetwork
from pmrf.modeling import NetworkSystem


VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")
# Add a method to the logger for convenience
def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)
logging.Logger.verbose = verbose

logger = logging.getLogger(__name__)


class NetworkFitterSettings:
    """
    Settings for a circuit fitter. This includes settings for the solver, the ports of the models to fit against, the frequency range etc.
    The recommended way to initialize all these settings is simply by passing them as kwargs directly to the "NetworkFitter" class or sub-class.
    """
    def __init__(self, **kwargs):
        # Input settings
        self.data_path = None                                                           # The absolute path to input data.        

        # Output settings
        self.title = 'test'                                                             # The file prefix to use for output files.
        self.output_path = 'output'                                                     # The base path that all outputs are relative to. Set to None to have no output.
        self.append_title = False                                                       # Whether or not to append the above title to the output path i.e. to set output_path = output_path + title
        self.silent = False
        self.no_output = False
        self.init_logging = True
        self.log_level = 'verbose'                                                      # 'debug', 'verbose', 'info', 'warning' or 'error'
        self.save_every = 1000                                                          # When saving parameters, plots etc. during a fit, save every n function evaluations.

        # Frequency settings
        self.use_measured_frequency = False
        self.export_frequency = None

        # Default fitting settings
        self.target_ports = None                                                        # Port of the models to fit against, e.g. [(0,0), (1,0)]. The default of "None" means to fit against all the possible measured/model ports.

        # General solver settings
        self.solver = 'SLSQP'                                                           # Can be 'PolyChord' for a bayesian approach, or any of scipy optimize's solver types (typically e.g. 'SLSQP' or 'Nelder-Mead') for a frequentist approach.
        self.matrix_targets = 'free'                                                    # Either 'free' or 'all'. Defines which targets are used to calculate the feature matrix (and therefore both the cost and the likelihood functions).
        self.features = None                                                            # List of features to extract, namely 'real', 'imaginary', 'complex' or 'magnitude'. Default defined lower down.
        self.read_resume = False                                                        # Whether to resume from a previous run or not.
        self.save_settings = True                                                       # Whether to write a settings file containing these settings that can then be read later. Saved to the output path misc folder.

        # Bayesian solver settings (PolyChord)
        self.sigma_prior = UniformPDF(0.0, 0.015)
        self.parameter_method = 'likelihood-max'                                        # The method to choose a single 'best' parameter value. Can be 'likelihood-max' or 'param-mean'.
        self.num_live_points = None                                                     # The number of live points. Leave as None to use the number of circuit model parameters.

        # Frequentist solver settings (scipy.optimize)
        self.max_iterations = 1000                                                      # The maximum number of iterations before the solver terminates.
        self.cost_steps = ['L2', 'convolve-interleaved', 'L2', 'dB']                    # List of steps to perform for the total cost function (all targets e.g for the actual fit). See the Modifier class for options.
        self.cost_steps_individual = None                                               # List of steps to perform for the individual cost function (per target e.g for plot titles). Leave None to use same as self.cost_steps.

        # Target settings
        self.targets_free = None
        self.elements_free = None
        self.params_free = None
        
        self.update(kwargs)
        
            
    def update(self, d: dict):
        # The following populates settings from kwargs
        d_use = d.copy()
        for name, value in d.items():
            if hasattr(self, name):
                setattr(self, name, value)
                d_use.pop(name)

        # The following enforces constraints between settings
        if self.features is None:
            if self.solver == 'PolyChord':
                self.features = ['real', 'imaginary'] # currently only either 'real' and 'imaginary' (Gaussian likelihood) or 'magnitude' (Rician likelihood) are supported
            else:
                self.features = ['complex', 'magnitude']
                
        if self.silent == True:
            self.init_logging = False
            self.no_output = True
                    
    def to_dict(self):
        def recurse(obj):
            if isinstance(obj, rf.Frequency):
                return frequency_to_dict(obj)
            if hasattr(obj, '__dict__'):
                return {k: recurse(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [recurse(item) for item in obj]
            else:
                return obj
            
        return recurse(self)

    @classmethod
    def from_dict(cls, d):
        def recurse(obj, d):
            for k, v in d.items():
                if v is None:
                    pass
                elif isinstance(obj.__dict__.get(k, None), rf.Frequency):
                    v = dict_to_frequency(v)
                elif hasattr(getattr(obj, k, None), '__dict__'):
                    v = recurse(getattr(obj, k), v)
                elif isinstance(v, list):
                    # setattr(cls_or_obj, k, [recurse(type(item)(), item) if isinstance(item, dict) else item for item in v])
                    # v = [recurse(type(item)(), item) for item in v if item is not None]
                    v = [recurse(type(item)(), item) if isinstance(item, dict) else item for item in v]
                # elif isinstance(v, dict):
                    # v = {item_k: recurse(type(item_v)(), item_v) for item_k, item_v in v.items() if item_v is not None}
                
                setattr(obj, k, v)
            
            return obj

        obj = cls()
        return recurse(obj, d)


class NetworkFitter:
    """
    A network fitter is a class consisting of some fitting targets that dependent on some parametric network elements
    as well as some measured data. Ultimately, the main purpose of the Fitter is to fit RF model parameters
    to measured data.

    It has been designed such that all individual elements and targets can be disabled
    in the fit (i.e. held constant) or replaced by some measured data. It also has various utility functions,
    such as the plotting of parameters and data.
    
    When deriving from a Fitter, one should implement `init_targets()`, which populates self.target with a list of Target objects.
    Otherwise, the default implementation enables fits on a single model and measured network,
    passed in as "model" and "measured" upon creation. If you want to initialize your own models,
    simply don't pass model and measured, and override these functions appropriately.
    """
    def __init__(self, models: NetworkSystem | list[ParametricNetwork], measured: list[rf.Network] = None, settings = None, load_settings = False, **kwargs):
        """The initializer for a NetworkFitter.

        Args:
            models (NetworkSystem | list[ParametricNetwork]): The models to fit against. Can be a complex NetworkSystem, or simply a list of parametric networks.
            measured (list[rf.Network], optional): A list of measured networks. Defaults to None, which is useful for derived classes, where measurements should the n be populated in _init_targets().
            settings (_type_, optional): A setting struct to initialize settings from. Generally key-word arguments are passed instead.
            load_settings (bool, optional): Specifies whether or not to load settings from file, in which case only the key-word argument "output_path" need be passed.
            **kwargs: Key-word arguments. This is the main way to configure the class. Possible arguments are all members of the NetworkFitterSettings class.
        """
        # Settings
        self._init_settings(settings=settings, load_settings=load_settings, **kwargs)       
       
        # Models and measured data
        if isinstance(models, list):
            if self._settings.use_measured_frequency:
                kwargs['frequency'] = measured[0].frequency
            system = NetworkSystem(models, **kwargs)
        else:
            system = models
            
        self._system: NetworkSystem = system
        self._measured: list[rf.Network] = measured
        self._targets: list[Target] = []
        
        # Cost and likelihood setup
        self._cost_chain_combined = ModifierChain(self._settings.cost_steps)
        self._cost_chain_individual = ModifierChain(self._settings.cost_steps_individual or self._settings.cost_steps)
        self._likelihood_object = None
        self._likelihood_priors = {}
        
        # Other
        self._fit_output = None
        self._plotter = None
        
        # All other initialization wrapped in try except
        output_dir_exists = self.output_path is not None and Path(self.output_path).exists()
        try:
            if not self._settings.output_path is None:
                self._init_output()

            if self._settings.init_logging:
                if rank == 0:
                    self._init_logging()
                else:
                    self._init_logging(level='info')

            logger.info(f"Creating Fitter (random ID = {uuid.uuid4()})")
            logger.verbose(f"Python args: {' '.join(sys.argv)}")
            logger.verbose(f"Title: '{self._settings.title}'")
            logger.verbose(f"Solver: {self._settings.solver}")
            logger.verbose(f"Features: {self._settings.features}")
          
            logger.verbose("Initializing targets")
            self._init_targets()
            self._init_params()
            
            if self._settings.targets_free is not None:
                self.set_free_targets(self._settings.targets_free)
            if self._settings.elements_free is not None:
                self.set_free_elements(self._settings.elements_free)
            if self._settings.params_free is not None:
                self.set_free_params(self._settings.params_free)
            
            self._init_plotting()
            
            if self._settings.save_settings and not self.output_path is None:
                self.save_settings()

        except Exception as e:
            if not output_dir_exists:
                shutil.rmtree(Path(self.output_path))
            raise e
        
    def _init_settings(self, settings=None, load_settings=False, save_settings=False, **kwargs):
        self._settings: NetworkFitterSettings = settings or NetworkFitterSettings(**kwargs)        
        
        if load_settings:
            append_title_original = self._settings.append_title
            if not Path(self.output_settings_path).exists():
                self._settings.append_title = True # try again with appending the title
            
            if Path(self.output_settings_path).exists():
                logger.verbose("Loading fitter settings from file")
                self.load_settings()
                self._settings.update(kwargs)
                self._settings.read_resume = True
            else:
                self._settings.append_title = append_title_original
                raise Exception(f'Load settings requested but path {self.output_settings_path} does not exist')          
                
        # Don't override existing settings unless save_settings has been explicitly passed
        if load_settings and not save_settings:
            self._settings.save_settings = False        
        
    def _init_output(self):
        if not self._settings.export_frequency is None:
            raise Exception('Currently experiencing errors when exporting to a different frequency due to a bug in ComputableNetwork interpolate_self()')        
            self._export_frequency = self._settings.export_frequency
        
        Path(self.output_path).mkdir(exist_ok=True, parents=True)
        Path(self.output_param_path).mkdir(exist_ok=True, parents=True)
        Path(self.output_misc_path).mkdir(exist_ok=True, parents=True)
        
        np.save(f'{self.output_misc_path}/frequency.npy', self.export_frequency.f)
        
    def _init_logging(self, level=None):
        log_level = level or self._settings.log_level
        if log_level == 'debug':
            level = logging.DEBUG
        elif log_level == 'verbose':
            level = VERBOSE
        elif log_level == 'info':
            level = logging.INFO
        elif log_level == 'warning':
            level = logging.WARNING
        elif log_level == 'error':
            level = logging.ERROR

        if not self._settings.output_path is None:
            logging.basicConfig(filename=f'{self.output_misc_path}/out.log', level=level)
        else:
            logging.basicConfig(level=level)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        logging.getLogger().addHandler(stream_handler)
        
    def _init_params(self, param_set=None):               
        logger.verbose("Initializing parameters")

        # Initialize bayesian likelihood parameters
        if self.is_bayesian:
            sigma = self._settings.sigma_prior.mean
            if self._settings.features in [['real', 'imaginary'], ['imaginary', 'real'], ['real'], ['imaginary']]:
                self._likelihood_object = GaussianLikelihood(sigma=sigma)
                self._likelihood_priors = {'sigma': self._settings.sigma_prior}
            elif self._settings.features == ['complex']:
                self._likelihood_object = CircularComplexGaussianLikelihood(sigma_real=sigma, sigma_imag=sigma)
            elif self._settings.features == ['magnitude']:
                self._likelihood_object = RicianLikelihood(sigma=sigma)
                self._likelihood_priors = {'sigma': self._settings.sigma_prior}
            else:
                raise Exception(f'No likelihood available for selected features of type {self._settings.features}')
            
            logger.verbose(f'Likelihood type: {self._likelihood_object.kind()}')
        else:
            logger.verbose(f'Cost steps: {self._settings.cost_steps}')            
        
        # If read resume was passed, and we have params to load from, update the active params
        if self._settings.read_resume:
            logger.verbose('Resuming from previous parameters')
            loaded = False

            # Don't do this on init anymore
            # if self.is_bayesian:
            #     try:
            #         _ = self.nested_samples
            #         self.update_params_from_samples()
            #         loaded = True
            #     except:
            #         pass
            
            prev_param_path = f'{self.output_param_path}/opt.csv'
            if not loaded and Path(prev_param_path).is_file():
                self._system.load_params(file=f'{self.output_param_path}/opt.csv')
        
        # Save the initial params to file
        self._system.save_params(f'{self.output_param_path}/initial.csv')
        
    def _init_plotting(self):
        if rank == 0:
            logger.verbose("Initializing plotting")
            self._plotter = self.make_plotter()
        else:
            self._plotter = None

    def _init_targets(self):
        """
        Method that sub-classes should override, used to initialize targets.
        
        Default behaviour is to use the features within the settings, as well as the individual cost chains,
        to initialize the targets. Sub-classes may therefore simply initialized self._measured and call this function.
        if they do not wish to initialize self._targets themselves. Additional behaviour is also implemented
        to attempt to load measured data from the input path.
        
        The targets do NOT have to be frequency-correct: the Fitter class will set their frequencies appropriately.
        """
        if len(self._targets) != 0:
            return
        
        # Load measured data if not already done so
        if not self._measured:
            try:
                self._measured = [rf.Network(self.input_data_path)]
            except:
                files = glob.glob(f'{self.input_data_path}/*.s[0-9]*p')
                if len(files) == 0:
                    raise Exception('Could not find any touchstone files in the input data path')
                else:
                    self._measured = [rf.Network(files[0])]
        if not self._measured:
            raise Exception('No measured data provided')
        
        # Update frequency settings
        # if not self._settings.use_measured_frequency:
        #     for measured in self._measured:
        #         measured.interpolate_self(self.models.frequency)
        
        # Features and target populations
        for i, model in enumerate(self._system.networks):
            features = []
            measured = self._measured[i]
            
            model_port_tuples = model.port_tuples
            measured_port_tuples = measured.port_tuples
            target_ports = self._settings.target_ports or list(set(model_port_tuples).intersection(measured_port_tuples))
            
            for ports in target_ports:
                features.extend([Feature(feature, ports=ports) for feature in self._settings.features])
        
            target = Target(model, measured=measured, features=features, cost_chain=self._cost_chain_individual)
            self._targets.append(target)

    @property
    def settings(self) -> NetworkFitterSettings:
        return self._settings

    @property
    def output_path(self) -> str:
        if self._settings.append_title:
            return f'{self._settings.output_path}/{self._settings.title}'
        else:
            return f'{self._settings.output_path}'

    @property
    def output_misc_path(self) -> str:
        return f'{self.output_path}/misc'
    
    @property
    def output_settings_path(self) -> str:
        return f'{self.output_misc_path}/settings.json'

    @property
    def output_touchstone_path(self) -> str:
        return f'{self.output_path}/touchstone'

    @property
    def output_param_path(self) -> str:
        return f'{self.output_path}/params'

    @property
    def output_polychord_path(self) -> str:
        return f'{self.output_path}/polychord'
    
    @property
    def polychord_file_root(self) -> str:
        return 'polychord'
    
    @property
    def chains_root(self) -> str:
        return f'{self.output_polychord_path}/{self.polychord_file_root}'

    @property
    def input_data_path(self) -> str:
        return self._settings.data_path
    
    @property
    def system(self) -> NetworkSystem:
        return self._system

    @property
    def frequency(self) -> rf.Frequency:
        return self.system.frequency
    
    @property
    def export_frequency(self) -> rf.Frequency:
        return self._settings.export_frequency or self.system.frequency
    
    @property
    def num_model_params(self):
        return len(self.system.params.names_free)
    
    @property
    def num_likelihood_params(self):
        if not self.is_bayesian:
            return 0
        else:
            return self._likelihood_object.num_params
    
    @property
    def nested_samples(self):
        if not self.is_bayesian:
            raise Exception('Nested samples are only available for bayesian solvers')
        return read_chains(self.chains_root)   
    
    @property
    def samples(self):
        params = self.system.params.names_free
        nested_samples = self.nested_samples
        return nested_samples.loc[:, params].to_numpy()       
    
    @property
    def weights(self):
        nested_samples = self.nested_samples
        return nested_samples.get_weights()
    
    @property
    def plotter(self) -> Plotter:
        return self._plotter
    
    @property
    def targets(self) -> list[Target]:
        return self.targets.copy()
    
    @property
    def targets_free(self) -> list[Target]:
        return [target for target in self._targets if not target.fixed]
    
    @property
    def is_bayesian(self) -> bool:
        if self._settings.solver == 'PolyChord':
            return True
        else:
            return False    
    
    def make_plotter(self, free_only=True) -> Plotter:
        if free_only:
            targets = [target for target in self._targets if not target.fixed]
        else:
            targets = self._targets
        
        plotter = Plotter(targets, self._system)
        self.update_plotter(plotter)
        return plotter

    def update_plotter(self, plotter=None):
        if plotter is None:
            plotter = self._plotter

        nested_samples = None
        if self.is_bayesian:
            plotter.bayesian = True
            try:
                nested_samples = self.nested_samples
            except:
                pass

        output_path = f'{self.output_path}/figures'
        plotter._likelihood_object = self._likelihood_object
        plotter.is_bayesian = self.is_bayesian
        plotter.no_output = self._settings.no_output
        plotter.params = self.system.params
        plotter.output_path = output_path
        plotter._nested_samples = nested_samples        
        if self._settings.no_output:
            plotter.save = False

    def network_from_target(self, target: Target, theta=None, noise=False) -> rf.Network:
        self.update_params(theta, update_noise=noise)
        return target.model

    def target_from_name(self, name) -> Target:
        target = None
        for target_search in self._targets:
            if target_search.name == name:
                target = target_search
                break

        if target == None:
            raise Exception(f'Target {name} not found')

        return target

    def targets_from_names(self, target_names) -> list[Target]:
        if target_names is None:
            return self._targets

        targets = []
        for name in target_names:
            targets.append(self.target_from_name(name))

        return targets

    def set_free_targets(self, target_names: list[str] = None):
        if target_names is None:
            for target in self._targets:
                target.fixed = False
        else:
            for target in self._targets:
                if target.name in target_names:
                    target.fixed = False
                else:
                    target.fixed = True                    
        self.reset_params()

    def set_free_elements(self, element_names: list[str] = None):
        if element_names is None:
            for element in self.elements:
                element.fixed = False
        else:
            for element in self.elements:
                if element.name in element_names:
                    element.fixed = False
                else:
                    element.fixed = True        
        self.reset_params()

    def set_free_params(self, param_names: list[str] = None):
        if param_names is None:
            self._params_original.fixed = False
        else:
            self._params_original.fixed = True
            self._params_original.loc[self._params_original.index.isin(param_names), 'fixed'] = False        
        self.reset_params()

    def feature_matrix(self, features=None, separate=False, ignore_imag=False) -> np.ndarray:
        if self._settings.matrix_targets == 'free':
            targets = [target for target in self._targets if not target.fixed]
        elif self._settings.matrix_targets == 'all':
            targets = self._targets
        else:
            raise Exception('Unknown feature targets setting')

        feature_extractors: list[list[Feature]] = []
        measured_networks: list[rf.Network] = []
        target_networks: list[rf.Network] = []

        for target in targets:
            target_networks.append(target.model)
            measured_networks.append(target.measured)
            if features is None:
                features_list = target.features_list
            else:
                features_list = [Feature(f) for f in features]
            feature_extractors.append(features_list)

        y_meas = extract_features(measured_networks, feature_extractors, ignore_imag=ignore_imag)
        y_target = extract_features(target_networks, feature_extractors, ignore_imag=ignore_imag)

        if separate:
            return y_meas, y_target
        else:
            return y_meas - y_target

    def cost(self, theta=None, reset_params=False, features=None) -> np.float64:
        reset_params = reset_params and not theta is None
        if reset_params:
            x_before = self.system.params.values()
        if not theta is None:
            self.update_params(theta)

        y = self.feature_matrix(features=features)
        cost = self._cost_chain_combined(y)
        
        if reset_params:
            self.update_params(x_before)        
            
        return cost
    
    def log_likelihood(self, theta=None, reset_params=False) -> np.float64:
        reset_params = reset_params and not theta is None
        if reset_params:
            x_before = self.system.params.values()
        if not theta is None:
            self.update_params(theta)        
            
        y_meas, y_target = self.feature_matrix(separate=True, ignore_imag=True)
        logL = self._likelihood_object(y_meas, y_target)
        
        if reset_params:
            self.update_params(x_before)
            
        return logL
    
    def __call__(self, theta):
        # Allows to easily use this object as a callable, e.g. for scipy.optimize or polychord, or data analysis packages
        if self.is_bayesian:
            return self.log_likelihood(theta=theta)
        else:
            return self.cost(theta=theta)        
    
    def log_evidence(self):
        return self.nested_samples.logZ()
    
    def update_networks(self):
        self._system.update_networks()    

    def fit_params(self, plotter='default', reset_params=False):
        """
        Fit the parameters, meaning find their best values (frequentist solvers) or determine their posteriors (bayesian solvers).
        """
        if plotter == 'default':
            plotter = self.plotter

        if reset_params:
            self.reset_params()
        else:
            self.update_networks()

        target_names = [target.name for target in self._targets if not target.fixed]
        param_names = self.system.params.names_free

        logger.verbose(f'Free Targets: {target_names}')
        logger.verbose(f'Free Parameters: {param_names}\n')
        logger.verbose(f'Fitting for {self.system.num_free_params} circuit model parameter(s)...')

        start = time.time()
        
        if self.is_bayesian:            
            retval = self._fit_params_bayesian()
            success = True
        else:
            retval = self._fit_params_frequentist(plotter=plotter)
            success = retval.success
        
        end = time.time()
        elapsed = timedelta(seconds=end - start)
        if success:
            logger.verbose(f'Fit complete successfully in {elapsed}\n')
        else:
            logger.error(f'Fit complete with errors! solver message: {retval.message}\n')
                
        self.save_params()
        self.update_plotter()

        return retval

    def reset_params(self):
        self._system.reset_params()

    def update_params(self, params: np.ndarray | dict, update_networks=True, update_fitter_likelihood=True, update_network_likelihoods=True, update_noise=False, scaler=None):
        if isinstance(params, dict):
            raise Exception('Updating parameters directly from dict not yet supported')

        params_networks = params[0:self.num_model_params]
        if not self.is_bayesian:
            update_fitter_likelihood = False
            update_network_likelihoods = False
            params_likelihood = None
        else:
            params_likelihood = params[-self.num_likelihood_params:]

        if update_fitter_likelihood:
            self._likelihood_object.update_params(params_likelihood)

        if update_networks:
            self._system.update_params(params_networks, scaler=scaler)
            for target in self._targets:
                target.update_params(params_likelihood, update_noise=update_noise, update_likelihoods=update_network_likelihoods)
                
    def update_params_from_samples(self):
        if not self.is_bayesian:
            raise Exception('Can only update parameters from samples if solver is Bayesian')

        param_names = self.system.params.index[self.system.params.fixed == False]
        param_names_replaced = [name.replace('_', '.') for name in param_names]
        param_names = np.array([[name, f'\\theta_{{{name_replaced}}}'] for name, name_replaced in zip(param_names, param_names_replaced)])                

        # Update the active parameters based on the desired method.
        try:
            nested_samples = self.nested_samples
            if self._settings.parameter_method == 'param-mean':
                # Network params
                for param in param_names[:,0]:
                    value = nested_samples[param].mean()
                    self.system.params.loc[param, 'value'] = value
            elif self._settings.parameter_method == 'likelihood-max':
                idx = np.argmax(nested_samples.logL.values)
                for param in param_names[:,0]:
                    value = nested_samples[param].values[idx]
                    self.system.params.loc[param, 'value'] = value
            else:
                raise Exception('Unknown best parameter method')
            
            
        except:
            pass
        
        self.update_params()
       
    def save_params(self, title='opt'):
        if rank != 0:
            return
        self._system.save_params(f'{self.output_param_path}/{title}.csv')

    def save_touchstone(self):
        if rank != 0:
            return
        self._system.save_touchstone(self.output_touchstone_path, networks=True, subnetworks=True, export_frequency=self.export_frequency)
            
    def load_settings(self):
        with open(self.output_settings_path, 'r') as f:
            settings_dict = json.load(f)
            if self._settings is None:
                settings_created = NetworkFitterSettings.from_dict(settings_dict)
            else:
                settings_created = type(self._settings).from_dict(settings_dict)

            if self._settings is None:
                self._settings = settings_created
            else:
                self._settings.update(settings_created.__dict__)

            # Old code
            # settings_created = NetworkFitterSettings.from_dict(settings_dict)
            # if self._settings is None:
            #     self._settings = settings_created
            # else:
            #     self._settings.update(settings_created.__dict__)
    
    def save_settings(self):
        if rank != 0:
            return
        
        logger.verbose("Saving fitter settings to file")

        with open(self.output_settings_path, 'w') as f:
            json.dump(self._settings.to_dict(), f, indent=4, default=lambda o: o.__dict__)                    

    def _fit_params_bayesian(self):
        """
        Fit the parameters using a bayesian approach.
        Currently this uses polychord, with a uniform prior from the minimum to maximum parameter values.
        Then, the mean of the posterior's is used as the updated parameter values, as well as the posterior's plotted.
        """
        # Get param names
        model_param_names = list(self.system.params.index[self.system.params.fixed == False])
        likelihood_param_names = list(self._likelihood_object.params().keys())
        
        # Populate latex param names
        param_names = model_param_names + likelihood_param_names
        param_names_replaced = [name.replace('_', '.') for name in param_names]
        param_names = np.array([[name, f'\\theta_{{{name_replaced}}}'] for name, name_replaced in zip(param_names, param_names_replaced)])

        # Setup likelihood and prior lambdas to pass to PolyChord
        callback_args = {
            'i_feval': 0,
        }
        
        likelihood = lambda theta: self._likelihood_callback(theta, callback_args=callback_args)
        prior = lambda hypercube: self._prior_callback(hypercube)

        # If there is a bug/crash in "likelihood" or "prior" above, we don't get useful error messages.
        # Uncomment the following code to make debugging easier.
        # _ = likelihood(self.models.params.value.to_numpy())
        # _ = prior(0.5 * np.ones(len(param_names)))
                
        self.system.params.enable_cache()

        # Run polychord. Useful parameters to investigate may be "precision_criterion" and "synchronous"
        kwargs = {
            'prior': prior,
            'paramnames': param_names,
            'read_resume': self._settings.read_resume,
            'base_dir': self.output_polychord_path,
            'file_root': self.polychord_file_root,
        }
        
        num_live_points = self._settings.num_live_points or self.system.num_free_params
        kwargs['nlive'] = num_live_points

        logger.verbose(f'Fitting for {self._likelihood_object.num_params} likelihood parameter(s)...')
        logger.info(f'PolyChord thread #{rank} started at {time_string()}')
        
        dumper = lambda _live, _dead, _logweights, logZ, _logZerr: logger.verbose(f'time: {time_string()} (logZ = {logZ:.2f})')

        self._fit_output = pypolychord.run(
            likelihood,
            len(param_names),
            dumper=dumper,
            **kwargs
        )

        self.system.params.flush_cache()

        self._plotter._nested_samples = self.nested_samples        
        self.update_params_from_samples()

        return self._fit_output

    def _fit_params_frequentist(self, plotter=None):
        """
        Fit the parameters using a frequentist approach.
        """
        # Get the active parameters and the prefix to use for output files
        params = self.system.params

        # Populate bounds and options
        x0 = params.loc[params.fixed == False, 'value'].to_numpy()
        priors = params.loc[params.fixed == False, 'pdf'].to_list()
        minimums = [prior.min for prior in priors]
        maximums = [prior.max for prior in priors]
        bounds = Bounds(minimums, maximums)
        options = {'maxiter': self._settings.max_iterations}

        # Setup the cost function lambda to pass to scipty
        callback_args = {
            'i_solver': 0,
            'i_feval': 0,
            'param_suffix': f'opt',
            'save_every': self._settings.save_every,
        }

        callback_args['plotter'] = plotter

        cost_callback = lambda x, callback_args : self._cost_callback(x, callback_args)
        def progress_callback(xk):
            callback_args['i_solver'] += 1

        # Run the minization routine
        # self.models.params.enable_cache()
        try:
            if self._settings.solver == 'shgo':
                self._fit_output = shgo(cost_callback, bounds=bounds, args=(callback_args), options=options, callback=progress_callback)
            else:
                self._fit_output = minimize(cost_callback, x0, args=(callback_args), bounds=bounds, method=self._settings.solver, options=options, callback=progress_callback)
        except KeyboardInterrupt:
            pass
        # self.models.params.flush_cache()

        return self._fit_output

    def _cost_callback(self, x, callback_args) -> float:
        try:
            self.update_params(x)
            cost = self.cost()
        except:
            cost = np.inf

        # The cost callback for frequentists. Update's self's parameters, calculates the cost, does some logging and then returns the cost
        callback_args['i_feval'] = callback_args['i_feval'] + 1

        save_every = callback_args['save_every']
        if save_every is not None and callback_args['i_feval'] % callback_args['save_every'] == 0:
            self.save_params()
            if callback_args['plotter'] is not None:
                callback_args['plotter'].plot_S(name='s_opt')

        logger.verbose(f"i_solver = {callback_args['i_solver']:5d},    i_feval = {callback_args['i_feval']:5d},    cost = {cost:.8f}")

        return cost

    def _likelihood_callback(self, theta, callback_args = None):
        # Update all parameters
        self.update_params(theta)

        try:
            logL = self.log_likelihood()
        except Exception as e:
            logL = -np.inf
            filename = f"error_{time_string()}.csv"
            logger.error(f"Likelihood function raised an exception. Active parameters saved to {filename}", exc_info=e)
            self.system.params.write_csv(f"{self.output_param_path}/{filename}")

        callback_args['i_feval'] = callback_args['i_feval'] + 1
        return logL

    def _prior_callback(self, hypercube):
        num_model_params = self.system.num_free_params
        
        model_values = [prior(hypercube[i]) for i, prior in enumerate(self.system.params.pdfs())]
        likelihood_values = [prior(hypercube[num_model_params + i]) for i, prior in enumerate(self._likelihood_priors.values())]
        
        return np.array(model_values + likelihood_values)
    