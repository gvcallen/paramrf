from typing import Any

from jaxtyping import PyTree, Array
import equinox as eqx

class InferenceResults(eqx.Module):
    """
    The high-level output of a parax sampling run.
    """
    #: The fully reconstructed parax model where parameters contain the posterior samples.
    #: Has a leading batch dimension of size (N_samples).
    model: PyTree
    
    #: The evaluated function values (log-likelihood or log-posterior) for each sample.
    fn_values: Array

    #: The statistical weights associated with each sample (if applicable).
    weights: Array | None = None

    #: Stacked auxiliary data from the physics evaluations.
    auxes: Any = None

    #: The raw backend-specific diagnostics/metrics object (e.g., blackjax.RMHState).
    metrics: PyTree