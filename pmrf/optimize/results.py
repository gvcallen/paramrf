import re
import logging
from dataclasses import dataclass

import numpy as np
import skrf
import h5py

from pmrf.results import BaseResults
from pmrf.models.model import Model
from pmrf.optimize.goal import Goal # Adjust import as needed

@dataclass
class OptimizeResults(BaseResults):
    """Container for the results of a goal-oriented design optimization."""
    goals: list[Goal] | None = None
    optimized_model: Model | None = None

    def _get_plot_data(self, use_initial_model=False, **kwargs) -> list[tuple]:
        """
        Prepares the optimized model's network and synthesizes "dummy" networks 
        to visualize the goal targets on the same skrf plots.
        """
        model = self.initial_model if use_initial_model else self.optimized_model
        if not model or not self.frequency:
            logging.warning("Missing model or frequency for plotting.")
            return []

        plot_data = []
        try:
            # 1. The Real Network
            opt_nw = model.to_skrf(self.frequency)
            opt_nw.name = "Optimized Model"
            
            networks = [opt_nw]
            plot_kwargs = [{'label': 'Optimized Model', 'linestyle': '-', 'color': 'b'}]

            # 2. Synthesize Target Networks from Goals
            if self.goals:
                n_freq = len(self.frequency)
                n_ports = opt_nw.number_of_ports
                target_added = False
                
                for i, goal in enumerate(self.goals):
                    # We only parse standard string aliases for plotting (e.g., 's11_db')
                    if not isinstance(goal.feature, str):
                        continue 
                        
                    match = re.match(r'^([a-zA-Z]+)(\d)?(\d)?(.*)$', goal.feature)
                    if not match: continue
                    
                    prefix, p1, p2, suffix = match.groups()
                    if p1 is None or p2 is None: continue
                    
                    m, n = int(p1)-1, int(p2)-1
                    if m >= n_ports or n >= n_ports: continue
                    
                    # Create an array of NaNs to avoid plotting lines on the wrong ports
                    s_target = np.full((n_freq, n_ports, n_ports), np.nan + 0j, dtype=complex)
                    
                    # Apply frequency masks if the goal only applies to a sub-band
                    T = goal.target
                    if goal.mask is not None:
                        T_arr = np.full(n_freq, np.nan)
                        # Ensure we extract numpy arrays for mask indexing
                        T_arr[np.array(goal.mask)] = T if isinstance(T, np.ndarray) else T
                        T = T_arr
                    elif isinstance(T, (int, float)):
                        T = np.full(n_freq, T, dtype=float)
                        
                    # Reverse-engineer the complex S-parameter so skrf plots the target correctly
                    if suffix == '_db':
                        s_target[:, m, n] = 10**(T/20) + 0j
                    elif suffix == '_mag':
                        s_target[:, m, n] = T + 0j
                    elif suffix == '_deg':
                        s_target[:, m, n] = np.cos(np.radians(T)) + 1j * np.sin(np.radians(T))
                    elif suffix == '_re':
                        s_target[:, m, n] = T + 0j
                    elif suffix == '_im':
                        s_target[:, m, n] = 0 + 1j * T
                    else:
                        s_target[:, m, n] = T + 0j
                        
                    # Build dummy skrf network for this specific goal
                    dummy_freq = skrf.Frequency.from_f(self.frequency.f, unit='Hz')
                    target_nw = skrf.Network(frequency=dummy_freq, s=s_target, name=f"Goal_{i}")
                    
                    networks.append(target_nw)
                    
                    # Style logic: Red for upper bounds (<), Green for lower bounds (>)
                    color = 'r' if goal.operator == '<' else 'g' if goal.operator == '>' else 'k'
                    label = f"Target ({goal.operator})" if not target_added else "_nolegend_"
                    target_added = True
                    
                    plot_kwargs.append({'label': label, 'linestyle': '--', 'color': color, 'linewidth': 2})

            plot_data.append((
                "Optimization Result", 
                networks, 
                plot_kwargs
            ))
        except Exception as e:
            logging.warning(f"Failed to generate optimized network for plotting: {e}")
                
        return plot_data

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