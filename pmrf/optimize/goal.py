from dataclasses import dataclass
from typing import Any
import jsonpickle
import jax.numpy as jnp

@dataclass
class Goal:
    """
    Defines a single design target for the optimizer.
    
    Parameters
    ----------
    feature : Any
        The feature to extract, using the exact syntax supported by 
        `pmrf.extract_features` (e.g., 's11_db', or {'lna': 'nf_db'}).
    operator : str
        The mathematical operator for the goal boundary: '<', '>', or '=='.
    target : float | jax.numpy.ndarray
        The target value. Can be a scalar for a flat goal across the band, 
        or an array of the same length as the frequency band.
    weight : float, default=1.0
        The relative importance of this goal.
    mask : jax.numpy.ndarray | None, default=None
        An optional boolean array to apply this goal only to specific 
        frequency points (e.g., passing sub-bands).
    """
    feature: Any
    operator: str
    target: float | jnp.ndarray
    weight: float = 1.0
    mask: jnp.ndarray | None = None

    def __post_init__(self):
        if self.operator not in ('<', '>', '=='):
            raise ValueError(f"Operator must be '<', '>', or '=='. Got '{self.operator}'")    
    
    def to_json(self) -> str:
        """Serialize the goal to a JSON string for HDF5 storage."""
        return jsonpickle.encode(self)
        
    @staticmethod
    def from_json(json_str: str) -> 'Goal':
        """Deserialize a goal from a JSON string."""
        return jsonpickle.decode(json_str)