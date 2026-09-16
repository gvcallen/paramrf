Parameter Names
===============

Every parameter in a model has a name, which acts as its key. Names are how parameters are read with :func:`pmrf.params` and :func:`pmrf.param_values`, changed with :func:`pmrf.update`, and tied together with :func:`pmrf.tie`. This page explains how names are formed, which units parameter values are expressed in, and what happens when a model is changed. For a hands-on walkthrough, see :doc:`/examples/parameter_naming_and_model_manipulation`.

.. _parameter-naming-rules:

How Names are Formed
~~~~~~~~~~~~~~~~~~~~

By default, a parameter is named by its attribute path through the model, such as ``cascade[0].R`` or ``substrate.dielectric.ep_r``. These paths are exact, but they change whenever the model is restructured. Giving models and parameters a ``name`` replaces the path with something more stable:

  * **Named models** act as a namespace. ``Resistor(50.0, name='r')`` gives ``r.R``, wherever the resistor ends up in a larger model. A named model at the root of the tree adds no namespace.
  * **Named parameters** replace their attribute path below the nearest named model, so ``prf.Unconstrained(1.0, name='C_global')`` is simply ``C_global``.
  * **Nested names** are joined with an underscore. An inductor named ``l1`` holding a parameter named ``L_val`` gives ``l1_L_val``.

This lets you choose the convention that suits your workflow: name only the parameters for a flat set of names, name only the models for namespaces, or mix the two. String dictionary keys that are valid Python identifiers are treated like attributes (``components.cable.length``), and any other key keeps its bracket form. Two parameters resolving to the same name is an error.

Wherever a name is accepted, a *selector* can be used instead: a glob such as ``'c.*'``, a sequence of names, or a callable that picks nodes out of the model. These functions also accept any collection of models, so a joint tree such as ``(model, noise_model)`` can be read or changed in a single call.

Parameter Spaces
~~~~~~~~~~~~~~~~

A parameter's value can be expressed in one of three spaces:

  * **Declared** space is the value as written, in the units given by the parameter's scale. ``Capacitor(prf.Unconstrained(2.0, scale=1e-12))`` has a declared value of ``2.0``.
  * **Physical** space is the scaled SI value, ``2e-12`` here. This is what the model's equations see.
  * **Raw** space is the unbounded value that an optimizer or sampler moves through. For a bounded parameter, its constraint maps the value onto the real line.

Declared space is the default everywhere. Construction, bounds, priors, a model's ``repr``, :func:`pmrf.params`, :func:`pmrf.update` and :func:`pmrf.derivative` all use it, so what you read back is what you wrote. The other spaces are reached by passing ``space='physical'`` or ``space='raw'``. Reading and writing are inverses in every space, which means the raw values an optimizer returns can be written straight back into the model.

Since a scale defines the units a value is written in, a scale given explicitly on a parameter replaces the model's default scale for that field rather than multiplying with it. Both the scale and the constraint also contribute a change-of-variables term to the prior density, so :func:`pmrf.log_prior` gives a different value in each space; its docstring describes each case.

Changing a Model
~~~~~~~~~~~~~~~~

Models are immutable, so :func:`pmrf.update` returns a changed copy rather than editing the model in place. The changes it makes fall into two kinds.

**Value changes**, such as passing a dictionary of new values or a ``value=`` argument, go through each parameter's constructor. The new value is checked against the parameter's bounds, and its prior, constraint, scale and name are all kept. Freeing or fixing parameters with ``fixed=`` works the same way, and only affects the parameters that are selected.

**Structural changes** replace a part of the model outright, either with a new sub-model or parameter, or with the result of a function applied to the old part. These put exactly what they are given in place, without validation. An exact name can select a whole sub-model here, whereas a glob only ever matches parameters.

The distinction matters for performance. RF methods such as :meth:`pmrf.Model.s` are compiled just-in-time, and the compiled code is only reused while the model's structure is unchanged. Changing a parameter's value keeps that structure, so it never triggers a recompile. Fixing or freeing a parameter, making a structural change, or swapping in a parameter with a different constraint or scale all change the structure, and so recompile. On a large circuit this can take noticeably longer than an evaluation, so value changes should be preferred inside loops.

Tied Parameters
~~~~~~~~~~~~~~~

Rather than setting a parameter once, :func:`pmrf.tie` derives it from another parameter. The target is removed from the model's parameters and is recomputed from its source every time the model is evaluated, so it follows the source through updates, optimization and sampling. Because it is no longer a parameter, it also no longer has a name. The tie function receives and returns physical values, and derivatives with respect to the source include the path through the tie.
