Core Primitives
===============

The core primitives in ParamRF are :class:`pmrf.Module`, :class:`pmrf.Model`, :class:`pmrf.Frequency`, and :class:`pmrf.Param`.

:class:`pmrf.Module` is the parameter-aware base class for immutable ParamRF containers. It provides parameter naming, functional updates, mapping, and tying without implying an RF response.

The Model
~~~~~~~~~
:class:`pmrf.Model` represents the base class for any RF model. All built-in models and components, such as :class:`~pmrf.models.components.lumped.Resistor`, :class:`~pmrf.models.components.lines.uniform.PhaseLine`, :class:`~pmrf.models.components.lines.uniform.CoaxialLine` etc. inherit from this class.

Under the hood, :class:`pmrf.Module` is an `Equinox Module <https://docs.kidger.site/equinox/api/module/module/>`_, a `JAX PyTree <https://docs.jax.dev/en/latest/pytrees.html>`_, and a Python `dataclass <https://docs.python.org/3/library/dataclasses.html>`_. :class:`pmrf.Model` inherits these properties. The practical consequences are:

  * **Modules are immutable**. They cannot be mutated in place. Use :meth:`pmrf.Module.at` to return an updated module instead.
  * **Models are lazy**. Since models only store their parameters (and not their S-matrix or frequency), they do not perform any computation at initialization. Rather, their response is only evaluated when one of their methods (e.g. :meth:`pmrf.Model.s`) is passed a :class:`~pmrf.Frequency` object *as input*. This is in contrast to other, purely object-oriented libraries (such as :mod:`scikit-rf`).
  * **Modules are JAX-native**. Modules are JAX `PyTrees <https://docs.jax.dev/en/latest/pytrees.html>`_, so they can be compiled, vectorized, and differentiated. See :doc:`jax_overview` for details.

For more information on how to combine and build custom models, see the :doc:`/examples/index` page.

Frequency, Parameters, and jnp.ndarray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :class:`pmrf.Frequency` class defines the axis over which models are evaluated. Ultimately, this is a lightweight wrapper around a JAX array (commonly imported as :class:`jnp.ndarray`). Those unfamiliar with JAX can see either the :doc:`jax_overview` section or have a look at JAX's own `quickstart <https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>`_ guide. However, for those seeking a TLDR: the API is very similar to NumPy's :class:`np.ndarray`, with a few "rough edges", for example `control flow <https://docs.jax.dev/en/latest/control-flow.html>`_ is handled quite differently.

All JAX arrays are treated as potential free parameters in ParamRF. However, in RF modeling, it is very common to want to specify parameter scaling, bounds, constraints or a prior probability distribution. To accomplish this, ParamRF exposes a parameter wrapper class :class:`pmrf.Param`. This class eagerly casts to a :class:`jnp.ndarray`, meaning that you can conveniently treat parameters as if they were regular arrays in your equations.

Power users may want to have a look at the `Parax <https://github.com/gvcallen/parax>`_ and `distreqx <https://lockwo.github.io/distreqx/>`_ documentation, which ParamRF uses for parameters and constraints under the hood.

The Evaluator and other classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~pmrf.evaluators.AbstractEvaluator` is an interface for a callable class that can be used to "evaluate" a model over frequency. Its output is a :class:`jnp.ndarray` (tensor or scalar). Evaluators can be used for extraction of model features, encapsulating loss/error/likelihood functions, and more. For example, to create a goal-oriented objective function for an optimization, the :class:`~pmrf.evaluators.Goal` evaluator can be used.

Evaluators are created automatically when fitting routines are called, such as :func:`~pmrf.optimize.minimize` or :func:`~pmrf.infer.sample`. For example, specifying ``'s21_db'`` as a fitting feature for :func:`~pmrf.fitting.fit` creates a :class:`~pmrf.evaluators.Feature` evaluator, whilst specifying a loss or likelihood function creates a :class:`~pmrf.evaluators.TargetLoss` or :class:`~pmrf.evaluators.MarginalLogLikelihood` evaluator respectively.

Since an evaluator takes frequency *as input*, the same evaluator can be re-used over different sweeps. For example, an evaluator can be fitted over a coarse sweep of measured data, and then plotted over a much finer one.

When a module is solved, an evaluator is paired with its sweep using :class:`pmrf.terms.BoundEvaluator`, which may also carry a relative weight. The result is a *term*: a callable taking the solved module. :class:`pmrf.problems.SummedTerms` combines that module with the terms evaluated over it.

Terms that do not require a frequency, such as a penalty on the model's parameters, can be written as a plain function of the model. The built-in terms inherit from :class:`pmrf.terms.AbstractTerm`, but this is a convenience and not a requirement. In most cases these classes do not need to be created manually, as :func:`~pmrf.optimize.minimize` and :func:`~pmrf.infer.sample` create them automatically.

Priors are parameter metadata, and are stripped when a tree is unwrapped for a solver, so they cannot be read from inside an evaluator or a term. Instead, :class:`pmrf.problems.PriorPenalized` extracts them once when it is constructed and evaluates them against the parameter values on each call. This is what separates a maximum a posteriori estimate from a maximum likelihood one, and it is selected automatically when `inference` is 'bayesian' in :func:`~pmrf.fitting.fit`.

Other core class interfaces include those in :class:`~pmrf.losses`, :class:`~pmrf.likelihoods`, :class:`~pmrf.discrepancy_models` and :class:`~pmrf.noise_models`. These help glue the rest of the library together, as well as enable more advanced features such as hyperparameter-based optimization and Gaussian process discrepancy modeling. See the :doc:`../tutorials/index` section for more information.
