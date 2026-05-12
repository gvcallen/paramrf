Core Classes
============

The two main classes in ParamRF are :class:`pmrf.Model` and :class:`pmrf.Frequency`, while the factories and classes in :mod:`pmrf.parameters` and :mod:`pmrf.evaluators` bring the library together.

The Model
~~~~~~~~~
:class:`~pmrf.Model` represents the base class for any RF model. All built-in models and components, such as :class:`~pmrf.models.components.lumped.Resistor`, :class:`~pmrf.models.components.lines.uniform.PhaseLine`, :class:`~pmrf.models.components.lines.uniform.CoaxialLine` etc. inherit from this class.

Under-the-hood, ParamRF uses the `Equinox <https://docs.kidger.site/equinox/api/module/module/>`_ library to interoperate with JAX. This means that :class:`~pmrf.Model` is an `Equinox Module <https://docs.kidger.site/equinox/api/module/module/>`_, a `JAX PyTree <https://docs.jax.dev/en/latest/pytrees.html>`_ and a Python `dataclass <https://docs.python.org/3/library/dataclasses.html>`_. If these concepts are completely foreign to you, do not worry. The practical consequences of this are:

  * Models are **immutable** and do not contain any state. Rather, they represent "pure functions" that simply have data and parameters attached. Although this takes some getting used to, the end result is better optimization performance, as well as differentiation capabilities.
  * To edit a model's parameters, ParamRF provides :meth:`pmrf.Model.at`. This provides an intuitive means of creating a new model with specific attributes changes, and is known at the **optics** or *lens* pattern in functional programming.
  * Since models are JAX PyTrees, any model methods are **JAX-native** and accept *JAX arrays*. This means that they can be compiled *Just-In-Time* for platforms enhanced performance, as well as platform such as GPUs, TPUs etc. Further, this opens the doors to many other advanced JAX/Equinox features, such as vectorization via :func:`jax.vmap` and differentiation via :func:`jax.jacfwd`. See the :doc:`jax_overview` section for more details.

To define custom models, you can inherit directly from :class:`~pmrf.Model`. When inherited from, methods such as :meth:`~pmrf.Model.s`, :meth:`~pmrf.Model.a`, :meth:`~pmrf.Model.z` and :meth:`~pmrf.Model.y` can be overridden to define model S-parameters, ABCD-parameters etc. as a function of frequency. This is a crucial distinction compared to other libraries (e.g. :mod:`scikit-rf`): a model **does not store its frequency**, but instead accepts its frequency as a function input. Then, any network properties that have not been manually overridden are automatically made available via RF conversion functions. These can also be found under :mod:`pmrf.rf`.

For more complex models, the :meth:`~pmrf.Model.__call__` method can also be overridden. Compared to the previous approach, :meth:`~pmrf.Model.__call__` does not accept any arguments as input, but instead must return a **fully constructed** :class:`~pmrf.Model` instance. This is very useful for declarative, hierarchical model building. For a deeper look into building and defining custom models, see the :doc:`model_building` section.

Frequency, parameters, and jax.Array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :class:`~pmrf.Frequency` class defines the axis over which models are evaluated. Ultimately, it is a lightweight wrapper around a JAX array (commonly imported as :class:`jnp.ndarray`). As mentioned, those unfamiliar with JAX can see either the :doc:`jax_overview` section in this documentation, or have a look at JAX's own `quickstart <https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>`_ guide for a more thorough overview. However, for those seeking a TLDR, the API is very similar to numpy's :class:`np.ndarray`, with some few "rough edges".

All JAX arrays are treated as potential free parameters in ParamRF. However, it is common to want to specify parameters with bounds, constraints or a prior probability distribution. To accomplish this, ParamRF exposes a number of parameter factories, constraints and distributions in :mod:`pmrf.parameters`, :mod:`pmrf.constraints` and :mod:`pmrf.distributions`. These parameters eagerly cast to a :class:`jnp.ndarray`, meaning that you can conveniently treat parameter's as if they were regular arrays in your equations.

Power users may be interested in the fact that, under-the-hood, parameters are created using the `Parax <https://github.com/gvcallen/parax>`_ and `distreqx <https://github.com/lockwo/distreqx>`_ libraries.

The Evaluator and other classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~pmrf.evaluators.AbstractEvaluator` is an interface for a callable class that can be used to "evaluate" a model over frequency. Its output is a :class:`jnp.ndarray` (tensor or scalar). Evaluators can be used for extraction of model features, encapsulating loss/error/likelihood functions, and more. For example, to create a goal-orientated objective function for an optimization, the :class:`~pmrf.evaluators.Goal` evaluator can be used.

Evaluators are created automatically when fitting routines are called, such as :func:`~pmrf.optimize.minimize` or :func:`~pmrf.infer.sample`. For example, specifying ``'s21_db'`` as a fitting feature for :func:`~pmrf.fitting.fit` creates a :class:`~pmrf.evaluators.Feature` evaluator, whilst specifying a loss or likelihood function creates a :class:`~pmrf.evaluators.TargetLoss` or :class:`~pmrf.evaluators.MarginalLogLikelihood` evaluator respectively.

Other core class interfaces include those in :class:`~pmrf.losses`, :class:`~pmrf.likelihoods`, :class:`~pmrf.discrepancy_models` and :class:`~pmrf.noise_models`. These help glue the rest of library together, as well as enable more advanced features such as hyperparameter-based optimization and Gaussian process discrepancy modeling. See the :doc:`../tutorials/index` section for more information.