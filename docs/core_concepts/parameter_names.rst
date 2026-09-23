Parameter Names
===============

Every parameter in a model has a name. Names are how parameters are read with :func:`pmrf.params` and :func:`pmrf.param_values`, changed with :func:`pmrf.update`, and tied with :func:`pmrf.tie`. For a hands-on walkthrough, see :doc:`/examples/parameter_naming_and_model_manipulation`.

.. _parameter-naming-rules:

How Names are Formed
~~~~~~~~~~~~~~~~~~~~

By default, a parameter is named by its attribute path through the model, such as ``cascade[0].R`` or ``substrate.dielectric.ep_r``. Paths change whenever the model is restructured, so models and parameters can be given a ``name`` instead:

  * **Named models** act as a namespace. ``Resistor(50.0, name='r')`` gives ``r.R``, wherever the resistor ends up in a larger model. A named model at the root of the tree adds no namespace.
  * **Named parameters** replace their attribute path below the nearest named model, so ``prf.Unconstrained(1.0, name='C_global')`` is ``C_global``.
  * **Nested names** are joined with an underscore. An inductor named ``l1`` holding a parameter named ``L_val`` gives ``l1_L_val``.

Name only the parameters for a flat set of names, only the models for namespaces, or mix the two. String dictionary keys that are valid Python identifiers are treated like attributes (``components.cable.length``), and any other key keeps its bracket form. Two parameters resolving to the same name is an error.

Wherever a name is accepted, a *selector* can be used instead: a glob such as ``'c.*'``, a sequence of names, or a callable that picks nodes out of the model. These functions also accept any collection of models, so a joint tree such as ``(model, noise_model)`` can be read or changed in a single call.

Parameter Spaces
~~~~~~~~~~~~~~~~

A parameter's value can be expressed in one of three spaces:

  * **Declared** space is the value as written, in the units given by the parameter's scale. ``Capacitor(prf.Unconstrained(2.0, scale=1e-12))`` has a declared value of ``2.0``.
  * **Physical** space is the scaled SI value, ``2e-12`` here. This is what the model's equations see.
  * **Raw** space is the unbounded value that an optimizer or sampler moves through. For a bounded parameter, its constraint maps the value onto the real line.

Declared space is the default everywhere. Construction, bounds, priors, a model's ``repr``, :func:`pmrf.params`, :func:`pmrf.update` and :func:`pmrf.derivative` all use it, so what you read back is what you wrote. The other spaces are reached by passing ``space='physical'`` or ``space='raw'``. Reading and writing are inverses in every space, so raw values from an optimizer can be written straight back.

A scale given on a parameter replaces the field's default scale; the two do not multiply. The scale and the constraint each add a change-of-variables term to the prior density, so :func:`pmrf.log_prior` differs between spaces (see its docstring).

Changing a Model
~~~~~~~~~~~~~~~~

Models are immutable, so :func:`pmrf.update` returns a changed copy. It makes two kinds of change.

**Value changes**, such as passing a dictionary of new values or a ``value=`` argument, go through each parameter's constructor. The new value is checked against the parameter's bounds, and its prior, constraint, scale and name are all kept. Freeing or fixing parameters with ``fixed=`` works the same way, and only affects the selected parameters.

**Structural changes** replace a part of the model outright, either with a new sub-model or parameter, or with the result of a function applied to the old part. The replacement is inserted as given, without validation. An exact name can select a whole sub-model here, whereas a glob only ever matches parameters.

A dictionary passed to :func:`pmrf.update` can hold both kinds. Each entry is decided by its value: a :class:`pmrf.Model` keyed by a sub-model name is a structural change, while an array or parameter keyed by a parameter name is a value change. For example, ``prf.update(system, {'east_coax': new_east, 'west_coax': new_west})`` replaces two sub-models in one call. A model given for a parameter name, or a value given for a sub-model name, raises an error.

RF methods such as :meth:`pmrf.Model.s` are JIT-compiled, and the compiled code is reused only while the model's structure is unchanged. Value changes keep the structure. Fixing or freeing a parameter, a structural change, or swapping in a parameter with a different constraint or scale all recompile, which on a large circuit can take much longer than an evaluation. Prefer value changes inside loops.

Tied Parameters
~~~~~~~~~~~~~~~

:func:`pmrf.tie` computes a parameter from another one. The target is removed from the model's parameters and is recomputed from its source every time the model is evaluated, so it follows the source through updates, optimization and sampling. Since it is no longer a parameter, it has no name. The tie function receives and returns physical values, and derivatives with respect to the source include the path through the tie.

Derived Models
~~~~~~~~~~~~~~

A tie can only relate parameters that already exist. Sometimes a relation needs a quantity the model has no place for. Take a coaxial cable of total length ``L`` that is wet for its first ``w``: the model is a wet section cascaded with a dry one, but neither section can hold ``L``, and ``L`` should keep its own prior (perhaps a joint lab prior with the cable's geometry) rather than drift as ``w`` changes.

:func:`pmrf.derived` turns a function ``f(base, **new)`` into a constructor. The resulting model holds the base model and the new parameters, and calls ``f`` on them whenever it is used.

.. code-block:: python

    @prf.derived
    def wet(cable, wet_length, wet_ep_r):
        wet = prf.replace(cable, length=wet_length,
                          dielectric=prf.replace(cable.dielectric, ep_r=wet_ep_r))
        dry = prf.replace(cable, length=cable.length - wet_length)
        return wet ** dry

    coax = wet(coax, wet_length=prf.Random(Uniform(0, 20), scale=1e-3),
               wet_ep_r=prf.Random(Uniform(1, 80)))

The result is an ordinary :class:`pmrf.Model`. Its parameters are the base's, under the same names, plus one per keyword (``wet_length`` and ``wet_ep_r``). The cable's geometry is used by both sections but is still one parameter, and a values dictionary saved from a fit of the dry cable applies unchanged. Nothing built inside ``f`` is named, and the derived model takes the base's name, so inside a named container everything is prefixed as usual.

Like a tie function, ``f`` receives physical values. Inside it, use :func:`pmrf.replace` to change fields of the object in hand, and :func:`pmrf.update` to change parts reached by name. ``f`` must be pure and must return a model whose structure does not depend on parameter values. It is part of the model's static structure, so define it once at module level: a lambda created anew on every call recompiles.

To share a parameter between several parts, derive at the level that owns them. Derived models nest and their names stay flat, so one water level for both arms of a balun is:

.. code-block:: python

    @prf.derived
    def wet_balun(system, wet_length):
        return prf.update(system, {
            'east_coax': wet(system.east_coax, wet_length=wet_length, wet_ep_r=80.0),
            'west_coax': wet(system.west_coax, wet_length=wet_length, wet_ep_r=80.0),
        })

    balun = wet_balun(balun, wet_length=prf.Random(Uniform(0, 20), scale=1e-3))

Here ``prf.params(balun)`` holds a single ``wet_length``, and changing it changes both cables. The reasoning behind this design is recorded in ADR-0003.
