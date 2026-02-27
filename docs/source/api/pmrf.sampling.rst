pmrf.fitting
============

The **sampling** module, with built-in sanmpling algorithms and results.

Samplers all receive the same base initialization arguments as :class:`BaseSampler <pmrf.fitting.BaseSampler>`,
as well as potential sample-specific arguments. Samplers are also categorized into either :class:`OneshotSampler <pmrf.sampling.OneshotSampler>` or :class:`AdaptiveSampler <pmrf.sampling.AdaptiveSampler>`,
and accept additional arguments accordingly.

To run the sampling, a single `run` function is provided. This runs the underlying sampling algorithm and backend.
Any additional key-word arguments are forwarded respectively.

Samplers
-----------
.. autosummary::
   :toctree: generated/sampling/samplers
   :caption: Submodules

   pmrf.sampling.UniformSampler
   pmrf.sampling.LatinHypercubeSampler
   pmrf.sampling.FieldSampler
   pmrf.sampling.EqxLearnSurrogateSampler

Results
-----------
.. autosummary::
   :toctree: generated/sampling/results
   :caption: Submodules

   pmrf.sampling.SampleResults

Bases
-----------
.. autosummary::
   :toctree: generated/sampling/bases
   :caption: Submodules

   pmrf.sampling.BaseSampler
   pmrf.sampling.OneshotSampler
   pmrf.sampling.AdaptiveSampler
   pmrf.sampling.SampleResults