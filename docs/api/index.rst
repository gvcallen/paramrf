API
=============

This page contains the main API reference.

Core Primitives
------------

.. autosummary::
   :toctree: generated/

   pmrf.Model
   pmrf.Frequency
   pmrf.Value
   pmrf.Param
   pmrf.Fixed
   pmrf.Bounded
   pmrf.Constrained
   pmrf.Random
   pmrf.param
   pmrf.field


Main Modules
-------------

.. autosummary::
   :toctree: generated/
   :recursive:

   pmrf.models
   pmrf.optimize
   pmrf.infer
   pmrf.fitting


Other Modules
-------------

.. autosummary::
   :toctree: generated/
   :recursive:

   pmrf.constraints
   pmrf.covariance_kernels
   pmrf.discrepancy_models
   pmrf.distributions
   pmrf.evaluators
   pmrf.likelihoods
   pmrf.losses
   pmrf.math
   pmrf.noise_models
   pmrf.parameters
   pmrf.rf
   pmrf.serialization
   pmrf.viz


Utilities
-----------------

.. autosummary::
   :toctree: generated/

   pmrf.NetworkCollection
   pmrf.Partial
   pmrf.InitVar
   pmrf.load
   pmrf.save
   pmrf.freeze
   pmrf.replace
   pmrf.unwrap
   pmrf.unwrap_self
   pmrf.as_param