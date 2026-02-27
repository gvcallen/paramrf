pmrf.fitting
============

The **fitting** module, with built-in fitting backends and results.

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

   pmrf.fitting.BlackJAXNSFitter
   pmrf.fitting.dyPolyChordFitter
   pmrf.fitting.NumPyroMCMCFitter
   pmrf.fitting.NumPyroNSFitter
   pmrf.fitting.OptaxFitter
   pmrf.fitting.PolyChordFitter
   pmrf.fitting.SciPyMinimizeFitter

Results
-----------
.. autosummary::
   :toctree: generated/fitting/results
   :caption: Submodules

   pmrf.fitting.AnestheticResults
   pmrf.fitting.NumPyroResults
   pmrf.fitting.SciPyResults

Bases
-----------
.. autosummary::
   :toctree: generated/fitting/bases
   :caption: Submodules

   pmrf.fitting.BaseFitter
   pmrf.fitting.FrequentistFitter
   pmrf.fitting.BayesianFitter
   pmrf.fitting.FitResults
   pmrf.fitting.FrequentistResults
   pmrf.fitting.BayesianResults
   pmrf.fitting.FitContext
   pmrf.fitting.FrequentistContext
   pmrf.fitting.BayesianContext