Working with Parameter Names
============================

Every parameter in a model has a name. Names are the key for saved results, for
selecting which parameters are free, and for handing parameters to external tools.
This page covers how names are formed, how to read and set parameters by name, and
how to flatten a model into a named vector.

Naming rules
~~~~~~~~~~~~

Names come from a single resolver, so a name returned by
:meth:`pmrf.Module.named_params` is accepted everywhere else, including
:meth:`pmrf.Module.at` and :meth:`pmrf.Module.tied`.

- **Attribute paths.** Without custom names, a parameter is named by its attribute
  path, e.g. ``feed.dielectric.ep_r``.
- **Named modules.** A module given ``name=`` collapses the path to its left into a
  namespace, so ``Resistor(R=..., name="load")`` held anywhere yields ``load.R``.
  Nested named modules are joined with ``namespace_separator`` (``_`` by default).
- **Named parameters.** A parameter given ``name=`` collapses its path to the
  nearest named module, or to the root.
- **Dictionary keys.** String keys that are valid Python identifiers become dotted
  names (``components.cable.length``). Other keys keep the bracket form
  (``params['a b']``).
- **Wrappers.** Names see through freezing and through wrappers such as
  :class:`pmrf.modules.Tied`, so they are relative to the wrapped module. A frozen
  parameter keeps its name but is not free.

.. code-block:: python

   import pmrf as prf
   from pmrf.models import Resistor, Capacitor
   from pmrf.distributions import Uniform

   class System(prf.Module):
       load: Resistor
       cap: Capacitor

   system = System(
       load=Resistor(R=prf.Random(Uniform(45.0, 55.0), value=50.0), name="load"),
       cap=Capacitor(C=prf.Bounded(0.5, 2.0, value=1.0, scale=1e-12)),
   )
   system.named_params()   # {'load.R': 50.0, 'cap.C': 1e-12}

Values and free sets
~~~~~~~~~~~~~~~~~~~~

:meth:`pmrf.Module.values` returns the physical (scaled) value of every parameter by
name, and :meth:`pmrf.Module.with_values` sets them. Everything else about each
parameter, its prior, constraint, scale and fixed state, is kept, so results can be
stored as a plain ``name -> value`` mapping and re-applied to a freshly built model.
Values outside a parameter's constraint raise.

.. code-block:: python

   values = system.values()
   system = system.with_values({"load.R": 51.0})

:meth:`pmrf.Module.with_free` makes exactly the parameters matching a set of
:mod:`fnmatch` globs free and freezes the rest. :meth:`pmrf.Module.with_fixed`
freezes the matches and leaves the rest alone.

.. code-block:: python

   only_load = system.with_free("load.*")
   no_cap = system.with_fixed("cap.*")

A single value can also be replaced on a parameter with
``prf.replace(param, value=...)``, which likewise takes the physical value.

Flattening to a vector
~~~~~~~~~~~~~~~~~~~~~~

External samplers and pipelines usually want a flat vector. :func:`pmrf.flatten`
gives one, along with its names, the way back to the model, and the log prior.
:func:`pmrf.optimize.run_minimizer` and :func:`pmrf.infer.run_sampler` are built on
the same function.

.. code-block:: python

   flat = prf.flatten(system, space="unconstrained")

   flat.names              # ('load.R', 'cap.C')
   flat.theta0             # 1-D array, aligned with flat.names
   flat.unflatten(theta)   # unwrapped model, ready for .s(freq)
   flat.wrap(theta)        # wrapped model with priors kept, e.g. for saving
   flat.log_prior(theta)   # scalar

The names are those of ``named_params(free_only=True)``, in the same order. An
array-valued parameter expands into one name per element in C order, so a
three-element ``coeffs`` gives ``coeffs[0]``, ``coeffs[1]`` and ``coeffs[2]``, and a
2-D ``w`` gives ``w[0,0]``, ``w[0,1]`` and so on.

Every method is a pure JAX function of ``theta``, so it works under ``jax.jit``,
``jax.grad`` and ``jax.vmap``:

.. code-block:: python

   import equinox as eqx
   import jax

   @eqx.filter_jit
   def log_posterior(theta):
       model = flat.unflatten(theta)
       return flat.log_prior(theta) + my_log_likelihood(model.s(freq))

   grad = jax.grad(log_posterior)(flat.theta0)

**Density space.** ``space`` selects what ``theta`` holds, and so which density
:meth:`~pmrf.FlatParams.log_prior` returns.

- ``'physical'``: ``theta`` holds the scaled values, as in :meth:`pmrf.Module.values`.
  Priors are authored on the unscaled value, and the scale is folded in, so the
  log prior is a density over the scaled values.
- ``'unconstrained'``: each value is mapped onto the real line through its
  constraint bijector. The log prior is a density over ``theta`` itself, and adds the
  log-determinant of the Jacobian of the map back to physical values:

  $$\log p_\theta(\theta) = \log p_x(f(\theta)) + \log\left|\det \frac{\partial f}{\partial \theta}\right|.$$

  This is the space gradient-based samplers such as NUTS expect.

Parameters without a prior contribute nothing, and joint priors attached with
:class:`pmrf.modules.Probabilistic` are included. The priors of fixed or frozen
parameters contribute a constant, matching :class:`pmrf.problems.PriorPenalized`.
