Parameter Naming and Model Manipulation
=========================================

All models store their parameters internally. Although it is usually easiest to set these parameters before you construct the model, it is sometimes convenient to change parameters or swap components already inside a model, or to place constraints over multiple parameters.

Since models are *immutable* and cannot reference each other (to align with JAX's requirements), parameters and sub-models cannot be edited directly (e.g., ``model.R = 50`` will fail), and also cannot point to the same objects in memory.

Instead, ParamRF provides free functions that take a model and return a new one: :func:`pmrf.params` and :func:`pmrf.param_values` to read parameters, :func:`pmrf.update` to change them, and :func:`pmrf.tie` to derive one from another. These functions accept parameter names or structural callables. How those names are formed is described in :ref:`parameter-naming-rules`, and the value spaces in :doc:`/core_concepts/parameter_names`.

Defining the Base Model
~~~~~~~~~~~~~~~~~~~~~~~

Let's define an RLC model with explicit names to demonstrate how naming and manipulation work together:

.. code-block:: python

  import pmrf as prf
  from pmrf.models import Resistor, Inductor, Capacitor, Short

  res = Resistor(R=100.0, name="myR")
  ind = Inductor(L=prf.Unconstrained(2.0, scale=1e-9, name="L_val"), name="myL")
  cap = Capacitor(C=prf.Unconstrained(1.0, scale=1e-12, name="C_global"))

  rlc = res ** ind ** cap

The model uses each of the naming rules from :ref:`parameter-naming-rules`: the resistor is a named model, the inductor is a named parameter inside a named model, and the capacitor is a named parameter only:

.. code-block:: python

  prf.params(rlc).keys()   # dict_keys(['myR.R', 'myL_L_val', 'C_global'])

Reading Parameters
~~~~~~~~~~~~~~~~~~

:func:`pmrf.param_values` returns the value of each parameter by name. Values are in the units each parameter declares, so the inductor reads back as ``2.0`` nH, exactly as it was written:

.. code-block:: python

  prf.param_values(rlc)                     # {'myR.R': 100.0, 'myL_L_val': 2.0, 'C_global': 1.0}
  prf.param_values(rlc, space='physical')   # {'myR.R': 100.0, 'myL_L_val': 2e-09, 'C_global': 1e-12}

The resistor was passed as a plain float, so it is fixed. To list only the parameters an optimizer may move, pass ``free_only=True``:

.. code-block:: python

  prf.params(rlc, free_only=True).keys()    # dict_keys(['myL_L_val', 'C_global'])

Modifying Specific Fields
~~~~~~~~~~~~~~~~~~~~~~~~~

Using :func:`pmrf.update`, we can change specific parameters and get back an updated model. Values are again given in the units each parameter declares.

You can pass a dictionary of values by name, or a selector (a single name, a glob, or a sequence of names) together with what to change:

.. code-block:: python

  # Updating a single value via its name
  rlc_R200 = prf.update(rlc, {"myR.R": 200.0})

  # Updating several values at once, in nH and pF
  rlc_tuned = prf.update(rlc, {"myL_L_val": 2.2, "C_global": 0.8})

  # Fixing multiple parameters simultaneously by passing a sequence of names
  rlc_fixed = prf.update(rlc, ["myL_L_val", "C_global"], fixed=True)

  # Freeing the resistor so it can be optimized too
  rlc_free = prf.update(rlc, "myR.R", fixed=False)

Changing values in this way keeps each parameter's bounds, prior and scale, and does not recompile the model's RF methods.

Because the model's structure is not rigid, we can also swap entire sub-models. A named model can be selected by its name, and replaced with any other model:

.. code-block:: python

  # Swapping the inductor out for a Short
  rlc_shorted = prf.update(rlc, "myL", Short())

The same form replaces a parameter with a new one, for example to give the capacitor bounds:

.. code-block:: python

  rlc_bounded = prf.update(
      rlc, "C_global", prf.Bounded(0.5, 2.0, value=1.0, scale=1e-12, name="C_global")
  )

Replacing a part in this way puts exactly what you pass in place, so the new parameter needs its own name and scale.

Tied Parameters
~~~~~~~~~~~~~~~

Sub-models and parameters can be *tied together* using :func:`pmrf.tie`. The tied result remains an RF model, so methods such as :meth:`pmrf.Model.s` remain available. You can define the target and source using parameter names or callables.

For example, to set the resistor's value to always be 100e12 times the capacitor's value:

.. code-block:: python

  # Using parameter names
  rlc_tied = prf.tie(
      rlc,
      target="myR.R",
      source="C_global",
      fn=lambda c: c * 100e12,
  )

  # Alternatively, using callables
  rlc_tied_func = prf.tie(
      rlc,
      target=lambda m: m.cascade[0].R,
      source=lambda m: m.cascade[1].cascade[1].C,
      fn=lambda c: c * 100e12,
  )

The tie function receives the source's physical value, in farads here, which is why it multiplies by 100e12. The resistor is no longer a parameter of the tied model; it is recomputed from the capacitor every time the model is evaluated, so it follows the capacitor through updates, optimization and sampling:

.. code-block:: python

  prf.params(rlc_tied).keys()                    # dict_keys(['myL_L_val', 'C_global'])
  rlc_tied = prf.update(rlc_tied, {"C_global": 2.0})   # the resistor is now 200 ohms

Note that a model built with ``**`` nests its cascades, which is why the callable above reaches the capacitor through ``cascade[1].cascade[1]``. Names do not depend on this structure, which makes them the more robust choice.

This provides a powerful API for more advanced optimization constraints.
