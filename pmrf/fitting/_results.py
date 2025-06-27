from typing import Any
from dataclasses import dataclass

from pmrf._model import Model

@dataclass
class FitResults:
    model: Model | None = None
    engine_results: Any = None

class FrequentistResults(FitResults):
    pass

class BayesianResults(FitResults):
    pass