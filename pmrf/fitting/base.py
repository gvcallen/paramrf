import glob
from abc import ABC, abstractmethod
import importlib
import logging
from typing import Sequence, Callable
from pathlib import Path

import matplotlib.pyplot as plt
import skrf
from skrf import Network

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.util import LevelFilteredLogger, iter_submodules, RANK
from pmrf.frequency import Frequency
from pmrf.constants import FeatureInputT
from pmrf import extract_features
from pmrf.network_collection import NetworkCollection
from pmrf.fitting.results import FitResults, FitSettings
from pmrf.fitting.context import FitContext

class BaseFitter(ABC):
    """
    An abstract base class that provides the foundation for all fitting algorithms in `pmrf`.
    """
    def __init__(
        self,
        model: Model,        
        *,
        features: FeatureInputT | None = None,
        output_path: str | None = None,
        output_root: str | None = None,
        sparam_kind: str = 'all',
    ) -> None:
        """
        Initializes the BaseFitter.

        Parameters
        ----------
        model : Model
            The parametric `pmrf` Model to be fitted.                                                                            
        features : FeatureInputT or None, optional
            Defines the features to be extracted from the model and network(s).
            Defaults to `None`, in which case real and imaginary features for all ports are used.
            Can be a single feature e.g. 's11', a list of features (e.g., `['s11', 's11_mag']`),
            or a dictionary with either of the above as value. In the dictionary case,
            keys must be network names in the collection passed by `measured` during fitting, which must also
            correspond to submodels which are attributes of the model. For example,
            {'name1', ('s11'), {'name2', ('s21')} can be passed.
            Note that if a collection of networks is passed, but a feature dictionary is not,
            it is assumed that those feature(s) should be extract for each networks/submodel.
            See `extract_features(..)` more details.
        output_path : str or None
            The path for fitters to write output data to. Defaults to `None`.
        output_root : str or None
            The root name to prepend (with an underscore) to output files in the output path. Defaults to `None`.
        sparam_kind : str or None
            The S-parameter data kind to use for port-expansion in feature extraction. Can either be 'transmission', 'reflection' or 'all'.
            See `extract_features` for more details.
        """
        # Populate parameters
        self.model: Model = model
        self.features: FeatureInputT | None = features
        self.output_path = output_path
        self.output_root = output_root
        self.sparam_kind = sparam_kind
        
        if RANK == 0:
            self.logger = logging.getLogger("pmrf.fitting")
        else:
            self.logger = LevelFilteredLogger(null_level=logging.WARNING)
            
    def fit(
        self,
        measured: str | Network | NetworkCollection,
        **kwargs         
    ) -> tuple[Model, 'FitResults']:
        """
        Fits the model to measured data.

        This method fits the full model using the features specified.

        Arguments are forwarded to ``self.run_context(...)``.

        Parameters
        ----------
        measured : skrf.Network or prf.NetworkCollection
            The measured network data to fit the model against.
            Can be a scikit-rf `Network` or a paramrf `NetworkCollection`.
            For the latter case the network names should be referenced during
            feature extraction by specifying features as a dictionary.
            If networks do not have the same frequency, a common frequency is used.
        **kwargs
            Additional arguments forwarded to the underlying algorithm via ``self.run_context``.

        Returns
        -------
        tuple[Model, FitResults]
            The fitted model and fit results.
        """
        if isinstance(measured, str):
            measured = skrf.Network(measured)
        
        ctx = self._create_context(measured)
        results = self._run_context(ctx, **kwargs)

        return results.fitted_model, results
    
    def fit_submodels(
        self,
        measured: NetworkCollection,
        **kwargs         
    ) -> tuple[Model, 'FitResults']:
        """
        Fits the submodels in a model to measured data.
         
        This method fits the model to the measured data by fitting its submodels in a sequential manner.

        Arguments are forwarded to ``self.run_context(...)``.

        Parameters
        ----------
        measured : prf.NetworkCollection
            The measured network data to fit the model against.
            Must be a ParamRF `NetworkCollection`. Network names should be referenced during
            feature extraction by specifying features as a dictionary.
            If networks do not have the same frequency, a common frequency is used.
        **kwargs
            Additional arguments forwarded to the underlying algorithm via ``self.run_context``.

        Returns
        -------
        tuple[Model, FitResults]
            The fitted model and fit results. `solver_results` in the fit results contains a dictionary of the individual submodel fit results.
        """
        all_results: dict[str, FitResults] = {}
        
        # Fit the components sequentially
        ctx_kwargs = kwargs.copy()
        ctx_kwargs['save_results'] = False
        ctx_kwargs['save_model'] = False
        ctx_kwargs.setdefault('figure_subfolder', '../../figures')

        for ntwk in measured:
            name = ntwk.name
            
            # Setup the submodel
            self.logger.info(f'Fitting {name}...')
            model = self.model.with_free_submodels([name], fix_others=True)
            comp_measured = measured.filter(lambda ntwk: ntwk.name == name)
            output_path = f'{self.output_path}/submodels/{ntwk.name}'
            
            # Run the fit
            ctx = self._create_context(comp_measured, model=model, output_path=output_path, output_root=name)
            comp_results = self._run_context(ctx, **ctx_kwargs)
            
            # Append the results
            all_results[name] = comp_results

        # Combine the models. We demote the parameter groups so that the belong to the submodel and not the parent model
        fitted_model = self.model.with_models([result.fitted_model for result in all_results.values()])
        fitted_model = fitted_model.with_param_groups_demoted()
        
        # Create the fit results and metadata
        fit_results = FitResults(
            initial_model=self.model,
            fitted_model=fitted_model,
            solver_results=all_results,

        )
        
        # Save the models and results
        if RANK == 0 and self.output_path is not None:
            self.logger.info(f'Saving combined model...')
            if kwargs.get('save_model', True):
                name = fitted_model.name or 'model'
                fitted_model.save(f'{self.output_path}/fitted_{name}.prf')
            self.logger.info(f'Saving combined results...')
            if kwargs.get('save_results', True):
                fit_results.save_hdf(f'{self.output_path}/fit_results.hdf5')

        return fit_results.fitted_model, fit_results

    def _create_context(self, measured, *, model=None, features=None, output_path=None, output_root=None, sparam_kind=None) -> FitContext:
        """
        Creates a FitContext from the provided measurement and optional overrides.

        Parameters
        ----------
        measured : skrf.Network or NetworkCollection
            The measured data.
        model : Model, optional
            Model override. Defaults to self.model.
        features : FeatureInputT, optional
            Features override. Defaults to self.features.
        output_path : str, optional
            Output path override. Defaults to self.output_path.
        output_root : str, optional
            Output root override. Defaults to self.output_root.
        sparam_kind : str, optional
            S-parameter kind override. Defaults to self.sparam_kind.

        Returns
        -------
        FitContext
            The initialized context object.
        """
        model = model or self.model
        features = features or self.features
        sparam_kind = sparam_kind or self.sparam_kind
        output_path = output_path or self.output_path
        output_root = output_root or self.output_root
        
        # Make sure measured is loaded, and that all frequencies are the same
        if isinstance(measured, str):
            measured = skrf.Network(measured)
        measured = measured.copy()
        if isinstance(measured, NetworkCollection):
            measured.interpolate_self()
            frequency = measured.common_frequency()
        else:
            frequency = measured.frequency
        frequency = Frequency.from_skrf(frequency)

        # Set the default features and ensure it is not a scalar
        features = features if features is not None else [port_feature for m, n in model.port_tuples for port_feature in (f's{m+1}{n+1}_re', f's{m+1}{n+1}_im')]
        if not isinstance(features, Sequence) and not isinstance(features, dict):
            features = [features]
        if isinstance(measured, NetworkCollection) and not isinstance(features, dict):
            features = {ntwk.name: features for ntwk in measured}

        measured_features = extract_features(measured, None, features, sparam_kind=sparam_kind)
        
        return FitContext(
            measured=measured,
            model=model,
            frequency=frequency,
            features=features,
            measured_features=measured_features,
            logger=self.logger,
            output_path=output_path,
            output_root=output_root,
            sparam_kind=sparam_kind,
        )
    
    def _run_context(
        self,
        context: FitContext,
        *,
        load_previous: bool = False, 
        new_uniform_frac: float | None = 0.1,
        save_model: bool = True,
        save_results: bool = True,
        plot: str | list[str] | None = 's_db',
        figure_subfolder: str | None = None,
        callback: Callable[['FitResults'], None] | None = None,
        **kwargs
    ) -> 'FitResults':
        """
        Executes the fitting context.

        This is a low-level method and should seldom be used directly.

        This method runs the fitting algorithm implemented by the underlying sub-class.
        It contains several convenience parameters, allowing for e.g. automatic saving
        and plotting of results.

        Additional arguments are forwarded to the underlying algorithm via ``self.run_algorithm``.

        Parameters
        ----------
        context: FitContext
            The fitting context.
        load_previous : bool, default=True
            Whether or not to try and load previous results from the output path.
        new_uniform_frac : float or None, optional, default=0.1
            The fraction to update model distribution bounds uniformly around the fitted model values.
        save_model : bool, default=True
            Saves the model to the output path (if provided).
        save_results : bool, default=True
            Saves the results to hdf format in the output path (if provided).
        plot : str | list[str] | None, default=['s_db']
            Features, such as S-parameters, to plot and save to the output path (if provided).
        callback : Callable[[FitResults], None] or None, optional
            A callback to run after fitting but before saving and plotting.
        **kwargs
            Additional arguments forwarded to the underlying fitter via ``self.run_algorithm``.

        Returns
        -------
        FitResults
            The fit results object.
        """
        # Try load from previous results
        if load_previous and context.output_path is not None:
            try:
                filename = glob.glob(f"{context.output_path}/*.hdf5")[0]
                results = FitResults.load_hdf(filename)
                logging.info(f"Loaded previous results.")
                return results
            except:
                pass

        # Output fit parameters and features
        self.logger.info(f"Fitting for {context.model.num_flat_params} parameters")
        self.logger.info(f"Parameter names: {context.model.flat_param_names()}")
        self.logger.info(f'Features: {context.features}')
        
        results = self._run_algorithm(context, **kwargs)
        results.measured = context.measured
        results.initial_model = context.model
        results.settings = context.settings(solver_kwargs=kwargs)

        if new_uniform_frac is not None:
            results.fitted_model = results.fitted_model.with_uniform_distributions(new_uniform_frac, respect_bounds=True)

        if callback:
            callback(results)
            
        if plot is not None and not isinstance(plot, list):
            plot = [plot]

        output_path = context.output_path
        save_output = context.output_path is not None and (save_model or save_results or plot is not None) and RANK == 0
        if save_output:
            output_prefix = f'{output_path}/{context.output_root}_' if context.output_root is not None else f'{output_path}/'
            fitted_model = results.fitted_model
            
            if save_model:
                Path(output_path).resolve().mkdir(parents=True, exist_ok=True)
                self.logger.info(f'Saving model...')
                fitted_model.save(Path(f'{output_prefix}fitted_model.prf').resolve())

            if save_results:
                Path(output_path).resolve().mkdir(parents=True, exist_ok=True)
                self.logger.info(f'Saving results...')
                results.save_hdf(Path(f'{output_prefix}results.hdf5').resolve())
        
            if plot is not None:
                self.logger.info(f'Plotting S-parameters...')
                if output_path is not None:
                    figure_path = f'{output_path}/{figure_subfolder}' if figure_subfolder is not None else output_path
                    figure_prefix = f'{figure_path}/{context.output_root}_' if context.output_root is not None else f'{figure_path}/'
                    Path(figure_path).resolve().mkdir(parents=True, exist_ok=True)
                else:
                    figure_path = None
                for plot_feature in plot:
                    func = getattr(results, f'plot_{plot_feature}')
                    func()
                    if figure_path is not None:
                        plt.savefig(Path(f'{figure_prefix}{plot_feature}.png').resolve(), dpi=400)
        
        return results
    
    @abstractmethod
    def _run_algorithm(self, context: FitContext, **kwargs) -> 'FitResults':
        """
        Executes the fitting algorithm.

        This is a low-level method and should seldom be used directly.

        This method must be implemented by all concrete subclasses. It is the
        main entry point to start the optimization or sampling process.

        Parameters
        ----------
        context : FitContext
            The fitting context.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        FitResults
            The fit results object.
        """        
        raise NotImplementedError    

def Fitter(
    model: Model,
    *,
    inference: str | None = None,
    backend: str | None = None,
    **kwargs
) -> 'BaseFitter':
    """
    Fitter factory function.
    
    This allows the creator of a fitter by simply specifying the inference type or fitter backend and having all arguments forwarded.
    See the relevant fitter classes for detailed documentation.

    Parameters
    ----------
    model : Model
        The parametric `pmrf` Model to be fitted.
        See the documentation for `BaseFitter`.
    inference : str, optional
        High-level inference mode. Can be either 'frequentist' or 'bayesian'.
        If provided and ``backend`` is ``None``, a suitable default backend
        is selected automatically.
    backend : str, optional
        Explicit fitter backend name. If provided, this takes precedence over
        ``inference`` and must be compatible with it.
    **kwargs
        Additional arguments forwarded to the fitter constructor.

    Returns
    -------
    BaseFitter
        The concrete fitter instance.
    """
    if inference is None and backend is None:
        inference = 'frequentist'
    if inference not in [None, 'frequentist', 'bayesian']:
        raise Exception('Unknown inference type')
    if backend is None:
        backend = 'scipy-minimize' if inference == 'frequentist' else 'polychord'
    
    if not is_inference_kind(backend, inference):
        raise Exception('Inference type incompatible with backend')

    cls = get_fitter_class(backend)
    return cls(model, **kwargs)

def is_frequentist(solver) -> bool:
    """
    Check if a solver is a Frequentist fitter.

    Parameters
    ----------
    solver : str
        The name of the solver.

    Returns
    -------
    bool
        True if the solver corresponds to a FrequentistFitter subclass.
    """
    from pmrf.fitting.frequentist import FrequentistFitter
    cls = get_fitter_class(solver)
    return issubclass(cls, FrequentistFitter)

def is_bayesian(solver) -> bool:
    """
    Check if a solver is a Bayesian fitter.

    Parameters
    ----------
    solver : str
        The name of the solver.

    Returns
    -------
    bool
        True if the solver corresponds to a BayesianFitter subclass.
    """
    from pmrf.fitting.bayesian import BayesianFitter
    cls = get_fitter_class(solver)
    return issubclass(cls, BayesianFitter)

def is_inference_kind(solver, inference: str):
    """
    Check if a solver matches a specific inference kind.

    Parameters
    ----------
    solver : str
        The name of the solver.
    inference : str
        The inference kind ('frequentist' or 'bayesian').

    Returns
    -------
    bool
        True if the solver matches the inference kind.

    Raises
    ------
    Exception
        If the inference kind is unknown.
    """
    if inference == 'frequentist':
        return is_frequentist(solver)
    elif inference == 'bayesian':
        return is_bayesian(solver)
    else:
        raise Exception(f"Unknown inference type '{inference}'")

def get_fitter_class(solver: str):
    """
    Retrieve the Fitter class corresponding to a solver name.

    Parameters
    ----------
    solver : str
        The name of the solver (e.g., 'scipy-minimize').

    Returns
    -------
    class
        The fitter class found in the backends.

    Raises
    ------
    Exception
        If the solver class cannot be found or imported.
    """
    solver = solver.replace('scipy', 'sciPy')
    solver = solver.replace('polychord', 'polyChord')

    class_names = [solver + 'Fitter']
    class_names.append(''.join(part[0].upper() + part[1:] for part in solver.split('-')) + 'Fitter')
    try:
        for submodule_name, _ in iter_submodules('pmrf.fitting._backends'):
            fitter_submodel = importlib.import_module(submodule_name)
            for class_name in class_names:
                if hasattr(fitter_submodel, class_name):
                    return getattr(fitter_submodel, class_name)
    except (ImportError, AttributeError) as e:
        raise Exception(f'Could not find solver named {solver} with error: {e}')
    
def fit(model: Model, measured: str | skrf.Network | NetworkCollection, **kwargs) -> tuple[Model, FitResults]:
    """Fits a model to measured data.

    This is an alternative API to ParamRF's :mod:`fitting <pmrf.fitting>` module.
    See the :func:`fit <pmrf.fitting.BaseFitter.fit>` method for more details.
    
    Internally, a fitter is first created using :func:`pmrf.fitting.Fitter`, and then :func:`fitter.fit(...)` is called,
    with the fitted model being returned. Fit results are stored in the model's metadata with key 'fit_results'.

    Key-word arguments are split into 'init' and 'fit' key-word arguments appropriately.

    Parameters
    ----------
    model: prf.Model
        The model to fit.
    measured : prf.NetworkCollection | skrf.Network | str
        The measured data.
    **kwargs
        Additional arguments forwarded to :func:`pmrf.fitting.Fitter` and :func:`pmrf.fitting.BaseFitter.fit`.

    Returns
    -------
    tuple[Model, FitResults]
        The fitted model and fit results.
    """        
    from pmrf.fitting import Fitter, FITTER_INIT_PARAMS
    init_kwargs = {k: kwargs.pop(k) for k in FITTER_INIT_PARAMS if k in kwargs}
    return Fitter(model, **init_kwargs).fit(measured, **kwargs)

def fit_submodels(model: Model, measured: str | skrf.Network | NetworkCollection, **kwargs) -> tuple[Model, FitResults]:
    """Fits a model's submodels to measured data.

    This is an alternative API to ParamRF's :mod:`fitting <pmrf.fitting>` module.
    See the :func:`fit_submodels <pmrf.fitting.BaseFitter.fit_submodels>` method for more details.
    
    Internally, a fitter is first created using :func:`pmrf.fitting.Fitter`, and then :func:`fitter.fit_submodels(...)` is called,
    with the fitted model being returned. Fit results are stored in the model's metadata with key 'fit_results'.

    Key-word arguments are split into 'init' and 'fit' key-word arguments appropriately.

    Parameters
    ----------
    model: prf.Model
        The model to fit.    
    measured : prf.NetworkCollection | skrf.Network | str
        The measured data.
    **kwargs
        Additional arguments forwarded to :func:`pmrf.fitting.Fitter` and :func:`pmrf.fitting.BaseFitter.fit_submodels`.

    Returns
    -------
    tuple[Model, FitResults]
        The fitted model and fit results. `solver_results` in the fit results contains a dictionary of the individual submodel fit results.
    """         
    from pmrf.fitting import Fitter, FITTER_INIT_PARAMS
    init_kwargs = {k: kwargs.pop(k) for k in FITTER_INIT_PARAMS if k in kwargs}
    return Fitter(model, **init_kwargs).fit_submodels(measured, **kwargs)