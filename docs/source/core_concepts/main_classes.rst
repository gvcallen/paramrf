Core Classes
============

All of the classes introduced below can be found in :mod:`pmrf.core`, but are exported at the root. These include :class:`pmrf.Model`, :class:`pmrf.Frequency`, :class:`pmrf.Evaluator`, and other helper classes for optimization and inference.

The Model
~~~~~~~~~
:class:`~pmrf.Model` represents the base class for any RF model. All built-in models and components, such as :class:`~pmrf.models.components.lumped.Resistor`, :class:`~pmrf.models.components.lines.uniform.PhaseLine`, :class:`~pmrf.models.components.lines.uniform.CoaxialLine` etc. inherit from this class.

The :class:`~pmrf.Model` class itself is a `parax.Module <https://gvcallen.github.io/parax/api/#parax.Module>`_, and therefore an `equinox.Module <https://docs.kidger.site/equinox/api/module/module/>`_, a `JAX PyTree <https://docs.jax.dev/en/latest/pytrees.html>`_ and a Python `dataclass <https://docs.python.org/3/library/dataclasses.html>`_. If these concepts sound foreign to you, do not worry. The practical consequences of this are:

  * Models are immutable, representing pure functions with attached data. You cannot update a model's parameters but modifying its attributes.
  * Parax are available on all models and should be used for model manipulation, such as freezing certain parameters with :meth:`pmrf.with_fixed_params()`, inspecting parameters with :meth:`pmrf.Model.named_params()`, and manipulating parameters with :meth:`pmrf.Model.with_params()`. See the `Parax documentation <https://gvcallen.github.io/parax/api/#parax.Module>`_ for more details.
  * All models are by default "JAX compatible". This allows for just-in-time compilation and computation on platforms such as GPUs, TPUs etc., as well as other advanced JAX/Equinox features (e.g. vectorization via :func:`jax.vmap` and differentiation via :func:`jax.jacfwd`).

To define custom models, you can inherit directly from :class:`~pmrf.Model`. When inherited from, methods such as :meth:`~pmrf.Model.s`, :meth:`~pmrf.Model.a`, :meth:`~pmrf.Model.z` and :meth:`~pmrf.Model.y` can be overriden to define model S-parameters, ABCD-parameters etc. as a function of frequency (a model does not store its frequency). Any methods that are not implemented manually are automatically made available via RF conversion which can be found in :mod:`pmrf.rf`.

For more complex models, the :meth:`~pmrf.Model.__call__` method can also be overridden. Compared to the previous approach, :meth:`~pmrf.Model.__call__` does not accept any arguments as input, but instead must return a fully constructed :class:`~pmrf.Model` instance. This is very useful for declarative, hierarchial model building. For a deeper look into composition and custom model building, see the :doc:`model_building` section.

Frequency, Parameter and jax.Array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :class:`~pmrf.Frequency` class provides a wrapper around a JAX array (referred to as :class:`jnp.ndarray`) that defines the frequency axis over which models are evaluated. Those unfamiliar with the JAX can see either the :doc:`jax_overview` section or JAX's own `quickstart <https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>`_ guide (TLDR: :class:`jnp.ndarray` is very similar to a regular numpy array, with a few exceptions). Frequency objects can represent either linear or arbitrarily spaced frequency values, but are usually instantiated using ``start``, ``stop``, and ``npoints`` and optionally a unit such as ``GHz``.

Since ParamRF builds on top of the `Parax <https://github.com/gvcallen/parax>`_ library, all parameters must be created using the `Parameter <https://gvcallen.github.io/parax/api/#parax.Parameter>`_ class. Similar to :class:`~pmrf.Frequency`, `parax.Parameter <https://gvcallen.github.io/parax/api/#parax.Parameter>`_ also wraps a JAX array, storing its value and additional metadata. This allows for parameter bounds and scaling, marking parameters as fixed, and associating a probability distribution with the parameter for Bayesian inference. However, unlike :class:`~pmrf.Frequency`, parameters eagerly cast to :class:`jnp.ndarray`, meaning you can treat them just like regular arrays or ``float`` variable. For more information, see the :doc:`model_building` page.

The Evaluator and other classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~pmrf.Evaluator` is a lower-level tool that can be used to "evaluate" a model over frequency in a composable manner and output an array (tensor or scalar) as a result. This can be used for extraction of model features, encapsulating loss/error functions, and more. For example, to create a goal-orientated objective function for an optimization, the :class:`~pmrf.evaluators.Goal` evaluator can be used. Evaluators are also Parax `Operators <https://gvcallen.github.io/parax/api/#parax.Operator>`_, meaning they can be added, subtracted, multiplied, negated etc.

Evaluators are created automatically when fitting routines are called, such as :func:`pmrf.optimize.fit` or :func:`pmrf.infer.condition`. For example, specifying ``'s21_db'`` as a fitting feature creates a :class:`~pmrf.evaluators.Feature` evaluator, whilst specifying a loss or likelihood function creates :class:`~pmrf.evaluators.TargetLoss` or :class:`~pmrf.evaluators.MarginalLogLikelihood` evaluators respectively. However, these can easily be created manually for more complex optimization or inference.

Other core classes include the :class:`~pmrf.Loss`, :class:`~pmrf.Likelihood`, :class:`~pmrf.DiscrepancyModel` and :class:`~pmrf.NoiseModel` classes. These help glue the rest of library together, as well as enable more advanced features such as hyper-parameter based optimization, and Gaussian process discrepancy modeling. See the :doc:`../tutorials/index` section for more information.