API
=============

This page contains the main API reference, providing a quick access section, as well as a detailed list of core classes and Python modules.

Quick Reference
---------------

.. rubric:: Models

* :mod:`~pmrf.models.adapters` (:mod:`~pmrf.models.adapters.static.Measured`, :mod:`~pmrf.models.adapters.callable.DiscreteCallable`, :mod:`~pmrf.models.adapters.callable.ContinuousCallable`, ...)
* :mod:`~pmrf.models.components` (:mod:`~pmrf.models.components.ideal`, :mod:`~pmrf.models.components.lines`, :mod:`~pmrf.models.components.lumped`, :mod:`~pmrf.models.components.nonideal`, :mod:`~pmrf.models.components.topological`, ...)
* :mod:`~pmrf.models.composite` (:mod:`~pmrf.models.composite.interconnected.Cascade`, :mod:`~pmrf.models.composite.interconnected.Circuit`, ...)
* :mod:`~pmrf.models.surrogates` (:mod:`~pmrf.models.surrogates.expansion.LinearExpansion`)

.. rubric:: Fitting, optimization and sampling
 
* :mod:`pmrf.fitting` (:func:`~pmrf.fitting.fit`, :func:`~pmrf.fitting.fit_sequential`)
* :func:`pmrf.optimize.minimize`
* :func:`pmrf.infer.sample`

.. rubric:: Serialization

* :func:`pmrf.load`
* :func:`pmrf.save`


Core Classes
------------

.. autosummary::
   :toctree: generated/

   pmrf.Model
   pmrf.Frequency
   pmrf.Evaluator
   pmrf.Problem
   pmrf.Loss
   pmrf.Likelihood
   pmrf.NoiseModel
   pmrf.DiscrepancyModel
   pmrf.CovarianceKernel


Modules
-------------

.. autosummary::
   :toctree: generated/
   :recursive:

   pmrf.discrepancy_models
   pmrf.evaluators
   pmrf.explore
   pmrf.fitting
   pmrf.infer
   pmrf.likelihoods
   pmrf.losses
   pmrf.math
   pmrf.models
   pmrf.network_collection
   pmrf.optimize
   pmrf.viz
   pmrf.rf
   pmrf.serialization