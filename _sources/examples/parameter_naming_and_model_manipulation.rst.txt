Parameter Naming and Model Manipulation
=========================================

Models are immutable, so a model's parameters cannot be edited in place (``model.R = 50`` will fail). Instead, ParamRF provides functions that read a model's parameters by name and return an updated copy. In this example, we build a named RLC network, then read its parameters, change them, swap out a component and tie two parameters together. For the rules behind names and value spaces, see :doc:`/core_concepts/parameter_names`.

Naming the Model
~~~~~~~~~~~~~~~~

Let's define an RLC network, giving names to some of its models and parameters:

.. plot::
   :context: reset
   :include-source:

   import pmrf as prf
   from pmrf.models import Resistor, Inductor, Capacitor

   res = Resistor(R=100.0, name='myR')
   ind = Inductor(L=prf.Unconstrained(2.0, scale=1e-9, name='L_val'), name='myL')
   cap = Capacitor(C=prf.Unconstrained(1.0, scale=1e-12, name='C_global'))

   rlc = res ** ind ** cap

   print(list(prf.params(rlc)))

**Output:**

.. code-block:: none

   ['myR.R', 'myL_L_val', 'C_global']

Each component uses a different naming style. The resistor is a named model, so its resistance is namespaced as ``myR.R``. The inductor is a named parameter inside a named model, so the two names are joined. The capacitor only names its parameter, so ``C_global`` stands on its own.

Reading Parameters
~~~~~~~~~~~~~~~~~~

:func:`pmrf.params` returns the :class:`pmrf.Param` objects themselves, whereas :func:`pmrf.values` returns just their values. Values are given in the units each parameter was declared in, so the inductor reads back as ``2.0`` nH. To get SI values instead, we can pass ``space='physical'``:

.. plot::
   :context:
   :include-source:

   prf.values(rlc)                     # {'myR.R': 100.0, 'myL_L_val': 2.0, 'C_global': 1.0}
   prf.values(rlc, space='physical')   # {'myR.R': 100.0, 'myL_L_val': 2e-09, 'C_global': 1e-12}

Both functions can also be given a selector, such as a glob, to pick out only some parameters. Since the resistor was passed as a plain float, it is fixed, and ``free_only=True`` leaves it out:

.. plot::
   :context:
   :include-source:

   print(list(prf.params(rlc, 'myL*')))
   print(list(prf.params(rlc, free_only=True)))

**Output:**

.. code-block:: none

   ['myL_L_val']
   ['myL_L_val', 'C_global']

Updating Parameters
~~~~~~~~~~~~~~~~~~~

To change parameter values, we pass a dictionary of new values to :func:`pmrf.update`, again in declared units. Each parameter keeps its bounds, prior and scale, and the model's RF methods do not need to be recompiled:

.. plot::
   :context:
   :include-source:

   rlc_tuned = prf.update(rlc, {'myL_L_val': 2.2, 'C_global': 0.8})   # 2.2 nH and 0.8 pF

A selector can also be used together with a keyword argument, for example to fix or free parameters. Here, a sequence of names fixes the inductor and capacitor, and a single name frees the resistor so that it can be optimized:

.. plot::
   :context:
   :include-source:

   rlc_fixed = prf.update(rlc, ['myL_L_val', 'C_global'], fixed=True)
   rlc_free = prf.update(rlc, 'myR.R', fixed=False)

Replacing Sub-models
~~~~~~~~~~~~~~~~~~~~

Rather than changing values, we can also replace whole parts of the model. A named model is selected by its name, so we can short out the inductor like this:

.. plot::
   :context:
   :include-source:

   from pmrf.models import Short

   rlc_shorted = prf.update(rlc, 'myL', Short())

   print(list(prf.params(rlc_shorted)))

**Output:**

.. code-block:: none

   ['myR.R', 'C_global']

The same approach replaces a parameter with a new one, for example to give the capacitor bounds. Since the new parameter is put in place exactly as given, it needs its own name and scale:

.. plot::
   :context:
   :include-source:

   rlc_bounded = prf.update(
       rlc, 'C_global', prf.Bounded(0.5, 2.0, value=1.0, scale=1e-12, name='C_global')
   )

Any later value update to ``C_global`` is now checked against these bounds.

Tying Parameters
~~~~~~~~~~~~~~~~

Finally, :func:`pmrf.tie` derives one parameter from another. Let's set the resistance to always be 100e12 times the capacitance. The target and source can be given by name, or by callables that select them from the model:

.. plot::
   :context:
   :include-source:

   # Using parameter names
   rlc_tied = prf.tie(rlc, target='myR.R', source='C_global', fn=lambda c: c * 100e12)

   # Using callables
   rlc_tied = prf.tie(
       rlc,
       target=lambda m: m.cascade[0].R,
       source=lambda m: m.cascade[1].cascade[1].C,
       fn=lambda c: c * 100e12,
   )

   print(list(prf.params(rlc_tied)))

**Output:**

.. code-block:: none

   ['myL_L_val', 'C_global']

The tie function receives the capacitance in farads, which is why it multiplies by 100e12. The resistance is no longer a parameter, and is instead recomputed from the capacitance whenever the model is evaluated. Updating the capacitor to 2 pF therefore sets the resistor to 200 ohms:

.. plot::
   :context:
   :include-source:

   rlc_tied = prf.update(rlc_tied, {'C_global': 2.0})

Note that the callables above reach the capacitor through ``cascade[1].cascade[1]``, since ``res ** ind ** cap`` nests one cascade inside another. Names do not depend on this structure, which makes them the more robust choice.
