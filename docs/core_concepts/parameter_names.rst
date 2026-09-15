Working with Parameter Names
============================

Every parameter in a model has a name. Names are the key for reading parameters,
for choosing which ones an optimizer may move, for tying one to another, and for
saved results. This page covers how names are formed, the spaces a value can be
read in, and how to change a model by name.

Throughout, ``rc`` is this two-component model:

.. code-block:: python

   import pmrf as prf
   from pmrf.models import Resistor, Capacitor

   rc = Resistor(50.0, name='r') ** Capacitor(prf.Unconstrained(2.0, scale=1e-12), name='c')

How names are formed
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   prf.params(rc).keys()   # dict_keys(['r.R', 'c.C'])

One resolver forms the names, so a name from :func:`pmrf.params` is accepted by
every other function on this page.

- **Attribute paths.** Without custom names, a parameter is named by its path
  through the model, such as ``cascade[0].R`` or ``substrate.dielectric.ep_r``.
- **Named modules.** A module given ``name=`` collapses the path to its left into
  a namespace, so ``Resistor(50.0, name='r')`` gives ``r.R`` wherever it is held.
  Nested named modules are joined with ``_``.
- **Named parameters.** A parameter given ``name=`` collapses its path to the
  nearest named module, or to the root.
- **Dictionary keys.** String keys that are valid Python identifiers give dotted
  names (``components.cable.length``); other keys keep the bracket form.

Names see through freezing and through wrappers such as
:class:`pmrf.modules.Tied`. The target of a tie is recomputed rather than stored,
so it has no name. Two parameters resolving to the same name raise.

Value spaces and scale
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   prf.param_values(rc)                     # {'r.R': 50.0, 'c.C': 2.0}
   prf.param_values(rc, space='physical')   # {'r.R': 50.0, 'c.C': 2e-12}

A parameter's number lives in one of three spaces:

- **declared**: the number as written, in the units the parameter's scale
  declares. ``2.0`` for 2 pF.
- **physical**: the scaled, SI value. ``2e-12``.
- **raw**: the latent, unbounded array an optimizer or sampler moves through. For
  a bounded parameter it is the value mapped onto the real line by its constraint.

Declared space is the default everywhere, and it is the space construction uses:
``Param(value=...)``, bounds, priors, ``repr`` and :func:`pmrf.params` all agree
with it. A scale is the units a value is written in, so an explicit scale on a
value overrides the field's default rather than multiplying with it:

.. code-block:: python

   from pmrf.models import Capacitor

   prf.param_values(Capacitor(prf.Unconstrained(2.0, scale=1e-9)), space='physical')  # 2e-9

:func:`pmrf.log_prior` takes the same ``space`` argument. Its docstring states the
measure each one gives, since the scale and the constraint each contribute a
change-of-variables term.

Reading parameters
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   prf.params(rc, 'c.*')                 # {'c.C': Param(variable=Real(...), scale=1e-12)}
   prf.params(rc, free_only=True).keys() # dict_keys(['c.C']): 'r.R' was passed as a float, so it is fixed

:func:`pmrf.params` returns :class:`pmrf.Param` objects, whose ``repr`` shows the
declared value, the bounds and the prior. :func:`pmrf.param_values` returns plain
arrays instead, which is the form optimizers take and :func:`pmrf.update` accepts.

Both take a **selector** as their second argument: a name, an :mod:`fnmatch` glob
over names, a sequence of names, or a callable returning nodes of the model. An
unknown name raises; a glob matching nothing selects nothing.

Both also work on any collection of models and parameters, not only a single
model, so a joint tree such as ``(model, noise_model)`` can be read in one call.

Changing a model
~~~~~~~~~~~~~~~~

.. code-block:: python

   rc = prf.update(rc, {'c.C': 3.0})   # 3 pF: values are in the units the parameter declares

:func:`pmrf.update` returns a copy of the model with the parts a selector picks
replaced. The form of the call says what replaces them:

.. code-block:: python

   prf.update(rc, 'c.*', value=3.0)              # one value for a selection
   prf.update(rc, 'r.*', fixed=False)            # free the resistor
   prf.update(rc, 'cascade[1]', Short())         # a new sub-model
   prf.update(rc, 'c.*', fn=lambda p: ...)       # a function of the old part
   prf.update(rc, values, space='raw')           # write-back from an optimizer

The mapping, ``value=`` and ``fixed=`` forms go through each parameter's
constructor, so a value is checked against the bounds and the prior, constraint,
scale, name and metadata are kept. The sub-model and ``fn=`` forms are structural:
they bypass validation and put exactly what they are given in place.

``fixed=`` is additive, and leaves parameters the selector does not match alone.
To free only some parameters, fix everything first:

.. code-block:: python

   rc = prf.update(prf.update(rc, '*', fixed=True), 'c.C', fixed=False)

Reading and writing are inverses, which is the round trip a fit relies on:

.. code-block:: python

   prf.update(rc, prf.param_values(rc, space='raw'), space='raw')   # the same model back

To derive one parameter from another instead of replacing it once, use
:func:`pmrf.tie`. The target is dropped from the model's parameters and
recomputed from the source every time the model is unwrapped, so it follows the
source through updates, optimization and sampling:

.. code-block:: python

   tied = prf.tie(rc, 'r.R', 'c.C', fn=lambda C: C * 5e13)
   prf.params(tied).keys()   # dict_keys(['c.C']): 'r.R' is derived, so it is no longer a parameter

What recompiles
~~~~~~~~~~~~~~~

RF methods such as :meth:`pmrf.Model.s` are compiled just-in-time, and the
compiled code is reused only while the model's structure is unchanged.

Changing a parameter's **value** never recompiles. The mapping, ``value=`` and
``space=`` forms of :func:`pmrf.update` keep the tree structure and every leaf's
dtype, shape and weak type, so the compiled ``s`` is reused:

.. code-block:: python

   import jax

   faster = prf.update(rc, {'c.C': 3.0})
   jax.tree_util.tree_structure(faster) == jax.tree_util.tree_structure(rc)   # True

**Rebuilding** a parameter does recompile, because it changes the structure. That
covers ``fixed=``, the structural forms, and constructing a new
:class:`pmrf.Param` with a different constraint or scale. This is expected and
usually what you want; it is worth knowing only because on a large circuit a
recompile can take noticeably longer than an evaluation. In a loop that sweeps a
value, use the value forms.
