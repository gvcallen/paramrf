API
=============

This page contains the main API reference, providing a quick access section, as well as a detailed list of core classes and Python modules.

Quick Reference
---------------

.. rubric:: Models

* :mod:`~pmrf.models.adapters` (:mod:`~pmrf.models.adapters.static.Measured`, :mod:`~pmrf.models.adapters.callable.DiscreteCallable`, :mod:`~pmrf.models.adapters.callable.ContinuousCallable`, ...)
* :mod:`~pmrf.models.components` (:mod:`~pmrf.models.components.ideal`, :mod:`~pmrf.models.components.lines`, :mod:`~pmrf.models.components.lumped`, :mod:`~pmrf.models.components.nonideal`, :mod:`~pmrf.models.components.topological`, ...)
* :mod:`~pmrf.models.composite` (:mod:`~pmrf.models.composite.interconnected.Cascade`, :mod:`~pmrf.models.composite.interconnected.Circuit`, ...)
* :mod:`~pmrf.models.surrogates` (:mod:`~pmrf.models.surrogates.rational.PoleResidue`, :mod:`~pmrf.models.surrogates.expansion.VectorExpansion`)

Core
------------

.. autosummary::
   :toctree: generated/

   pmrf.Model
   pmrf.Frequency
   pmrf.Param
   pmrf.param

Modules
-------------

.. autosummary::
   :toctree: generated/
   :recursive:

   pmrf.constraints
   pmrf.covariance_kernels
   pmrf.discrepancy_models
   pmrf.distributions
   pmrf.evaluators
   pmrf.fitting
   pmrf.infer
   pmrf.likelihoods
   pmrf.losses
   pmrf.math
   pmrf.models
   pmrf.noise_models
   pmrf.network_collection
   pmrf.optimize
   pmrf.parameters
   pmrf.viz
   pmrf.rf
   pmrf.serialization


Utilities
-----------------

.. autosummary::
   :toctree: generated/

   pmrf.Partial
   pmrf.NetworkCollection
   pmrf.load
   pmrf.save
   pmrf.combine
   pmrf.field
   pmrf.unwrap
   pmrf.as_fixed
   pmrf.as_frozen
   pmrf.as_free