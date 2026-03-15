import os
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

import matplotlib.pyplot as plt

from pmrf.models.model import Model
from pmrf.frequency import Frequency
from pmrf.runner import BaseRunner
from pmrf.constants import FeatureSpecT
from pmrf.io import save
from pmrf.util import RANK
from pmrf.optimize.goal import Goal
from pmrf.optimize.results import OptimizeResults

class BaseOptimizer(BaseRunner, ABC):
    r"""
    Base class for all ParamRF goal-oriented design optimizers.
    
    This runner optimizes model parameters to satisfy a set of specific 
    mathematical design goals (inequalities or exact targets).

    .. rubric:: Main methods

    .. autosummary::
       :nosignatures:
       
       run
       execute
    """
    def __init__(
        self,
        model: Model,
        goals: list[Goal],
        *,
        frequency: Frequency | None = None,
        **feature_kwargs
    ):
        self.goals = goals
        
        # 1. Parse the goals to set up the mathematical extraction problem
        features = [g.feature for g in goals]
        
        # 2. Let BaseRunner initialize the math graph (extract_features setup)
        super().__init__(
            model, 
            frequency=frequency, 
            features=features, 
            **feature_kwargs
        )

    def run(
        self,
        *,
        output_path: str | None = None,
        output_root: str | None = None,
        plot: FeatureSpecT | None = None,
        save_model: bool = True,
        save_results: bool = True,
        figure_dir: str | None = None,
        optimized_uniform_frac: float | None = None,
        **kwargs        
    ) -> tuple[Model, OptimizeResults]:
        r"""
        Execute the optimization scenario.
        """
        self.output_path = output_path
        self.output_root = output_root
        self.plot_features = plot
        
        if output_path is not None and RANK == 0:
            os.makedirs(output_path, exist_ok=True)        
        
        if self.frequency is None:
            raise ValueError("Frequency must be provided to the runner prior to optimization.")
            
        self.logger.info(f"Optimizing {self.model.num_flat_params} parameters against {len(self.goals)} goals")
        
        # 3. Execute Optimization (Backend doesn't need data arrays, it just calls self.cost)
        optimized_model, backend_results = self.execute(**kwargs)

        # 4. Package Results
        results = OptimizeResults()
        results.initial_model = self.model
        results.goals = self.goals
        results.optimized_model = optimized_model
        results.backend_results = backend_results
        results.backend_class = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        results.frequency = self.frequency
        results.features = self.features
        results.run_kwargs = kwargs

        # 5. Post-processing logic
        if optimized_uniform_frac is not None:
            results.optimized_model = results.optimized_model.with_uniform_distributions(
                optimized_uniform_frac, respect_bounds=True
            )

        if plot is not None and not isinstance(plot, list):
            plot = [plot]

        save_output = output_path is not None and (save_model or save_results or plot is not None) and RANK == 0
        plot_output = not save_output and RANK == 0
        if save_output:
            output_prefix = f'{output_path}/{output_root}_' if output_root is not None else f'{output_path}/'
            
            if save_model:
                Path(output_path).resolve().mkdir(parents=True, exist_ok=True)
                self.logger.info('Saving optimized model...')
                save(Path(f'{output_prefix}optimized_model.prf').resolve(), results.optimized_model)

            if save_results:
                Path(output_path).resolve().mkdir(parents=True, exist_ok=True)
                self.logger.info('Saving results...')
                results.save_hdf(Path(f'{output_prefix}optimize_results.hdf5').resolve())
        
            if plot is not None:
                self.logger.info('Plotting results...')
                figure_path = f'{output_path}/{figure_dir}' if figure_dir is not None else output_path
                figure_prefix = f'{figure_path}/{output_root}_' if output_root is not None else f'{figure_path}/'
                Path(figure_path).resolve().mkdir(parents=True, exist_ok=True)
                
                for plot_feature in plot:
                    func = getattr(results, f'plot_{plot_feature}')
                    func()
                    plt.savefig(Path(f'{figure_prefix}{plot_feature}.png').resolve(), dpi=400)
                    plt.close()
        elif plot_output:
            if plot is not None:
                for plot_feature in plot:
                    func = getattr(results, f'plot_{plot_feature}')
                    func()
        
        return results.optimized_model, results

    @abstractmethod
    def execute(self, **kwargs) -> tuple[Model, Any]:
        pass