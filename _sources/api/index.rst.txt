API
=============

This page contains the main API reference.

Core Primitives
---------------

.. autosummary::
   :toctree: generated/

   pmrf.Model
   pmrf.Module
   pmrf.Frequency
   pmrf.Param
   pmrf.unconstrained
   pmrf.fixed
   pmrf.bounded
   pmrf.constrained
   pmrf.random
   pmrf.uniform
   pmrf.normal
   pmrf.lognormal
   pmrf.truncated_normal

``bounded`` creates an interval constraint without a prior; ``uniform`` creates
a random parameter with a uniform prior. For ``normal``, ``lognormal`` and
``truncated_normal``, ``sigma`` is the distribution spread and ``scale`` is the
parameter's unit multiplier. ``lognormal`` takes ``loc`` and ``sigma`` in log
space.


Working with Parameters
-----------------------

.. autosummary::
   :toctree: generated/

   pmrf.params
   pmrf.values
   pmrf.log_prior
   pmrf.update
   pmrf.tie
   pmrf.resolve
   pmrf.derived


Main Modules
------------

.. autosummary::
   :toctree: generated/
   :recursive:

   pmrf.models
   pmrf.modules
   pmrf.materials
   pmrf.parameters
   pmrf.stats
   pmrf.objectives
   pmrf.optimize
   pmrf.infer
   pmrf.fitting

Other Modules
-------------

.. autosummary::
   :toctree: generated/
   :recursive:

   pmrf.math
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
