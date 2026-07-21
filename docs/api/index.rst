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
   pmrf.Problem
   pmrf.AbstractTerm
   pmrf.BoundEvaluator
   pmrf.NegativeLogPrior
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
   pmrf.terms
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
   pmrf.Attrgetter
   pmrf.Pathgetter
   pmrf.InitVar
   pmrf.is_param
   pmrf.as_param
   pmrf.param
   pmrf.partition
   pmrf.combine
   pmrf.batch_axes
   pmrf.batch_mask
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