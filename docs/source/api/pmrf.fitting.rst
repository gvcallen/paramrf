pmrf.fitting
============

.. automodule:: pmrf.fitting
    :members:
    :undoc-members:
    :show-inheritance:

The **fitting** module, with built-in fitting algorithms and results.

All fitters accept the same base initialization arguments as :class:`BaseFitter <pmrf.fitting.BaseFitter>`.
Additionally, fitters are categorized into either :class:`FrequentistFitter <pmrf.fitting.FrequentistFitter>` or :class:`BayesianFitter <pmrf.fitting.BayesianFitter>`,
and accept additional arguments accordingly.

To run the fits, several fitting routines are provided. The simplest is the :class:`fit <pmrf.fitting.BaseFitter.fit>`
method, which runs a single fit with the specified features and measured data. Key-word arguments are
forwarded to the specified fitter's `run` routine, which may provide additional run-time
arguments, and also generally forwards arguments to its specific backend (for example, PolyChord or SciPy).

Fitters
-----------
.. autosummary::
   :toctree: generated/fitting/fitters
   :caption: Submodules
   :recursive:

   BlackJAXNSFitter
   dyPolyChordFitter
   NumPyroMCMCFitter
   NumPyroNSFitter
   OptaxFitter
   PolyChordFitter
   SciPyMinimizeFitter

Results
-----------
.. autosummary::
   :toctree: generated/fitting/results
   :caption: Submodules
   :recursive:

   AnestheticResults
   NumPyroResults
   SciPyResults

Bases
-----------
.. autosummary::
   :toctree: generated/fitting/bases
   :caption: Submodules
   :recursive:

   BaseFitter
   FrequentistFitter
   BayesianFitter
   FitResults
   FrequentistResults
   BayesianResults
   FitContext
   FrequentistContext
   BayesianContext