import logging
from pmrf.models import Model
from pmrf.network_collection import NetworkCollection
from pmrf.fitting.base import BaseFitter
from pmrf.fitting.results import FitResults

def fit_sequential_submodels(
    model: Model, 
    measured: NetworkCollection,
    fitter_cls: type[BaseFitter],
    fitter_kwargs: dict | None = None,
    **run_kwargs
) -> tuple[Model, dict[str, FitResults]]:
    """
    A workflow to run independent, sequential fits of multiple submodels.
    
    Creates a dedicated fitter for each network in the measured collection.
    
    Parameters
    ----------
    model : Model
        The parent model containing the submodels.
    measured : NetworkCollection
        The collection of target data. The network names must match the submodel names.
    fitter_cls : type[BaseFitter]
        The Fitter class to instantiate for each submodel (e.g., SciPyMinimizeFitter).
    fitter_kwargs : dict, optional
        Arguments passed to the fitter's __init__ (e.g., cost_kind).
    **run_kwargs : dict
        Arguments passed to the fitter's .run() method (e.g., max_iterations).
    """
    fitter_kwargs = fitter_kwargs or {}
    all_results: dict[str, FitResults] = {}
    
    for ntwk in measured:
        name = ntwk.name
        logging.info(f'Fitting submodel: {name}')
        
        # 1. Isolate the submodel
        sub_model = model.with_free_submodels([name], fix_others=True)
        sub_measured = measured.filter(lambda n: n.name == name)
        
        # 2. Spin up a fresh, dedicated solver
        fitter = fitter_cls(sub_model, **fitter_kwargs)
        
        # 3. Run it and store results
        comp_results = fitter.run(sub_measured, **run_kwargs)
        all_results[name] = comp_results
        
    # 4. Stitch everything back together
    fitted_model = model.with_models([res.fitted_model for res in all_results.values()])
    fitted_model = fitted_model.with_param_groups_demoted()
    
    return fitted_model, all_results