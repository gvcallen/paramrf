import logging
from dataclasses import dataclass

import numpy as np
import h5py

from pmrf.results import BaseResults
from pmrf.models.model import Model
from pmrf import extract_features
from pmrf.optimize.goal import Goal # Adjust import as needed

@dataclass
class OptimizeResults(BaseResults):
    """Container for the results of a goal-oriented design optimization."""
    goals: list[Goal] | None = None
    optimized_model: Model | None = None

    def _get_feature_plot_data(self, feature: str, use_initial_model=False, **kwargs) -> list[dict]:
        """
        Extracts the requested feature from the model and superimposes 
        any matching goals as target lines.
        """
        model = self.initial_model if use_initial_model else self.optimized_model
        if not model or not self.frequency:
            logging.warning("Missing model or frequency for plotting.")
            return []

        plot_data = []

        # 1. Extract the actual feature from the model
        try:
            # extract_features returns shape (N_freq, N_features)
            y_model = extract_features(model, self.frequency, [feature])
            
            # extract_features returns complex arrays (even for _db), so we cast to real
            y_model = np.real(y_model[:, 0]) 
            
            plot_data.append({
                'y': y_model, 
                'label': 'Optimized Model', 
                'linestyle': '-', 
                'color': 'b',
                'linewidth': 1.5
            })
        except Exception as e:
            logging.warning(f"Failed to extract '{feature}' from model for plotting: {e}")

        # 2. Look for matching goals to overlay as target lines
        if self.goals:
            for i, goal in enumerate(self.goals):
                if goal.feature == feature:
                    n_freq = len(self.frequency)
                    T = goal.target
                    
                    # Create the target array
                    y_target = np.full(n_freq, np.nan)
                    
                    if goal.mask is not None:
                        # Apply target only where the boolean mask is True
                        mask_idx = np.array(goal.mask)
                        if isinstance(T, np.ndarray):
                            y_target[mask_idx] = T[mask_idx]
                        else:
                            y_target[mask_idx] = T
                    else:
                        y_target[:] = T
                        
                    # Color coding: Red for < (ceiling), Green for > (floor), Black for == (exact)
                    color = 'r' if goal.operator == '<' else 'g' if goal.operator == '>' else 'k'
                    
                    plot_data.append({
                        'y': y_target,
                        'label': f"Target ({goal.operator})",
                        'linestyle': '--',
                        'color': color,
                        'linewidth': 2
                    })

        return plot_data

    # --------------------------------------------------------------------------
    # I/O Functions (Unchanged)
    # --------------------------------------------------------------------------
    def _write_data(self, f: h5py.File):
        if self.optimized_model:
            self._write_model(f.create_group('optimized_model'), self.optimized_model)

        if self.goals:
            goals_grp = f.create_group('goals')
            for i, goal in enumerate(self.goals):
                goals_grp.attrs[f'goal_{i}'] = goal.to_json()

    @classmethod
    def _read_data(cls, f: h5py.File, kwargs: dict):
        if 'models' in f and 'optimized' in f['models']:
            kwargs['optimized_model'] = cls._read_model(f['models/optimized'])

        if 'goals' in f:
            goals_grp = f['goals']
            goal_keys = sorted([k for k in goals_grp.attrs.keys() if k.startswith('goal_')], 
                               key=lambda x: int(x.split('_')[1]))
            kwargs['goals'] = [Goal.from_json(cls._decode_str(goals_grp.attrs[k])) for k in goal_keys]