from dataclasses import dataclass
from typing import Dict, Any

from pmrf.core import Model

@dataclass
class OptimizeResult:
    model: Model
    cost: float
    history: Dict[str, Any]
    success: bool