from dataclasses import dataclass
from typing import Dict, Any

from pmrf.model import Model

@dataclass
class OptimizeResult:
    model: Model
    cost: float
    history: Dict[str, Any]
    success: bool