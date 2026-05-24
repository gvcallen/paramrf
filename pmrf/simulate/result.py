"""pmrf/simulate/result.py"""

from jaxtyping import ArrayLike
import equinox as eqx

from pmrf.simulate.base import AdmittanceResult, ScatteringResult
from pmrf.rf import y2s, s2y

class SimulateResult(eqx.Module):
    #: The underlying solution object.
    solution: AdmittanceResult | ScatteringResult
    
    #: The characteristic impedance to use to retrieve
    #: S-parameters for non-scattering result types.
    z0: ArrayLike = 50.0
    
    @property
    def s(self):
        if isinstance(self.solution, ScatteringResult):
            return self.solution.s
        elif isinstance(self.solution, AdmittanceResult):
            return y2s(self.solution.y, z0=self.z0)
        else:
            raise ValueError(f"Unknown underlying solution type: {type(self.solution)}")
            
    @property
    def y(self):
        if isinstance(self.solution, ScatteringResult):
            return s2y(self.solution.s, self.solution.z0)
        elif isinstance(self.solution, AdmittanceResult):
            return self.solution.y
        else:
            raise ValueError(f"Unknown underlying solution type: {type(self.solution)}")