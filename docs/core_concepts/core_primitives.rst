Core Primitives
============

The core primitives in ParamRF are the :class:`pmrf.Model` and :class:`pmrf.Frequency` classes, which are constructed using :mod:`~pmrf.parameters` and JAX arrays. :mod:`pmrf.evaluators` and other classes help bring the rest of the library together.

The Model
~~~~~~~~~
:class:`~pmrf.Model` represents the base class for any RF model. All built-in models and components, such as :class:`~pmrf.models.components.lumped.Resistor`, :class:`~pmrf.models.components.lines.uniform.PhaseLine`, :class:`~pmrf.models.components.lines.uniform.CoaxialLine` etc. inherit from this class.

Under the hood, ParamRF uses the `Equinox <https://docs.kidger.site/equinox/api/module/module/>`_ library to interoperate with JAX. This means that :class:`~pmrf.Model` is an `Equinox Module <https://docs.kidger.site/equinox/api/module/module/>`_, a `JAX PyTree <https://docs.jax.dev/en/latest/pytrees.html>`_ and a Python `dataclass <https://docs.python.org/3/library/dataclasses.html>`_. If these concepts are completely foreign to you, do not worry. The practical consequences of this are:

  * *Models are immutable*. Models cannot be mutated (``model.x = 5.0`` will throw an error). Rather, they represent "pure functions" which simply happen to store their parameters alongside. This means that models should never contain any fields that can be derived from other fields (initialization and function input should be separated), and should also not reference other objects in the model (using the same parameter object twice will result in two separate parameters!). Instead, :meth:`pmrf.Model.at` can be used to manipulate a model. Although this approach may feel relatively new for some, the end result is improved optimization performance, advanced differentiation capabilities, and more flexibility downstream.
  * *Models are lazy*. Since models only store their parameters (and not e.g. their S-parameter matrix or frequency), they do not perform any computation at initialization. Rather, their response is only evaluated when one of their methods (e.g. :meth:`pmrf.Model.s`) is passed a frequency object *as input*. This is in contrast to other, purely object-oriented libraries (such as :mod:`scikit-rf`).
  * *Models are JAX-native*. Models are simply JAX `PyTrees <https://docs.jax.dev/en/latest/pytrees.html>`_, meaning they can be passed around in any JAX context. This means that they can be compiled *just-in-time* (JIT) for enhanced performance and simulation on other platforms (GPUs, TPUs etc) and also used in many advanced JAX features (such as for vectorization via :func:`jax.vmap` and differentiation via :func:`jax.jacfwd`). See the :doc:`jax_overview` section for more details.

For more information on how to combine and build custom models, see the :doc:`/examples` page.

Frequency, Parameters, and jax.Array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :class:`~pmrf.Frequency` class defines the axis over which models are evaluated. Ultimately, this is a lightweight wrapper around a JAX array (commonly imported as :class:`jnp.ndarray`). Those unfamiliar with JAX can see either the :doc:`jax_overview` section or have a look at JAX's own `quickstart <https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>`_ guide. However, for those seeking a TLDR, the API is very similar to NumPy's :class:`np.ndarray`, with some few "rough edges" (e.g. `control flow <https://docs.jax.dev/en/latest/control-flow.html>`_ is handled differently).

All JAX arrays are treated as potential free parameters in ParamRF. However, in RF modeling, it is very common to want to specify parameter scaling, bounds, constraints or a prior probability distribution. To accomplish this, ParamRF exposes a number of parameter factories/constraints/distributions in :mod:`pmrf.parameters`, :mod:`pmrf.constraints` and :mod:`pmrf.distributions`. These parameters eagerly cast to a :class:`jnp.ndarray`, meaning that you can conveniently treat parameters as if they were regular arrays in your equations, but also carry their relevant metadata.

Power users may want to have a look at the `Parax <https://github.com/gvcallen/parax>`_ and `distreqx <https://lockwo.github.io/distreqx/>`_ documentation, which ParamRF uses under the hood.

The Evaluator and other classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~pmrf.evaluators.AbstractEvaluator` is an interface for a callable class that can be used to "evaluate" a model over frequency. Its output is a :class:`jnp.ndarray` (tensor or scalar). Evaluators can be used for extraction of model features, encapsulating loss/error/likelihood functions, and more. For example, to create a goal-oriented objective function for an optimization, the :class:`~pmrf.evaluators.Goal` evaluator can be used.

Evaluators are created automatically when fitting routines are called, such as :func:`~pmrf.optimize.minimize` or :func:`~pmrf.infer.sample`. For example, specifying ``'s21_db'`` as a fitting feature for :func:`~pmrf.fitting.fit` creates a :class:`~pmrf.evaluators.Feature` evaluator, whilst specifying a loss or likelihood function creates a :class:`~pmrf.evaluators.TargetLoss` or :class:`~pmrf.evaluators.MarginalLogLikelihood` evaluator respectively.

Other core class interfaces include those in :class:`~pmrf.losses`, :class:`~pmrf.likelihoods`, :class:`~pmrf.discrepancy_models` and :class:`~pmrf.noise_models`. These help glue the rest of the library together, as well as enable more advanced features such as hyperparameter-based optimization and Gaussian process discrepancy modeling. See the :doc:`../tutorials/index` section for more information.