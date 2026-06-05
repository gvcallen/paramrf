API
=============

This page contains the main API reference.

Core Primitives
---------------

.. autosummary::
   :toctree: generated/

   pmrf.Model
   pmrf.Frequency
   pmrf.Param
   pmrf.Unconstrained
   pmrf.Fixed
   pmrf.Bounded
   pmrf.Constrained
   pmrf.Random


Main Modules
------------

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
   pmrf.Bind
   pmrf.Attrgetter
   pmrf.InitVar
   pmrf.is_param
   pmrf.as_param
   pmrf.param
   pmrf.field
   pmrf.load
   pmrf.save
   pmrf.derivative
   pmrf.sweep
   pmrf.freeze
   pmrf.unfreeze
   pmrf.replace
   pmrf.unwrap
   pmrf.unwrap_self