Custom Models
=============

Sometimes, it is more convenient or elegant to define a custom model that is not available in ParamRF. For example, you may want to:

  * Define a model using a custom S-parameter equation
  * Implement a specialized version of an existing model
  * Add methods on top of your model e.g. for conversions/analysis

In ParamRF, this is done by inheriting directly from :class:`~pmrf.Model` and overriding at least one of its RF response methods, namely :meth:`~pmrf.Model.s`, :meth:`~pmrf.Model.a`, :meth:`~pmrf.Model.y`, :meth:`~pmrf.Model.z`, or :meth:`~pmrf.Model.primary_matrix`.

Defining a Capacitor
^^^^^^^^^^^^^^^^^^^^

Let's define a capacitor from first principles using its ABCD parameters:

.. plot::
   :context: reset
   :include-source:

   import pmrf as prf
   import jax.numpy as jnp
   
   class Capacitor(prf.Model):
       C: prf.Param
   
       def a(self, freq: prf.Frequency) -> jnp.ndarray:
           w, C = freq.w, self.C
           ones, zeros = jnp.ones_like(w), jnp.zeros_like(w)
   
           return jnp.array([
               [ones,  1.0 / (1j * w * C)],
               [zeros, ones]
           ]).transpose(2, 0, 1)

By inheriting from :class:`~pmrf.Model`, ``Capacitor`` becomes both a Python `dataclass <https://docs.python.org/3/library/dataclasses.html>`_ and a `JAX PyTree <https://docs.jax.dev/en/latest/pytrees.html>`_! For those familiar with dataclasses, this means that any standard dataclass syntax applies.

Note that :class:`~pmrf.Param` is merely a field *type-hint*, and does not enforce that the resulting field is registered as a parameter. To convert caller values into fixed or variable parameters returned by :meth:`pmrf.Model.named_params`, and to specify constraints and related metadata, use a field specifier.

Adding a Field Specifier
^^^^^^^^^^^^^^^^^^^^^^^^

ParamRF provides two field specifiers: :class:`~pmrf.field` and :class:`~pmrf.param`. For parameters, :class:`~pmrf.param` allows the following:

  * Setting a default value
  * Specifying parameter constraints that are inherent to the model
  * Attaching additional metadata and scaling at the model level
  * Auto-converting floats and other array-like values into variable/fixed parameters
  
The code below demonstrates this by extending the previous class, while constraining ``C`` to be only positive and defining the capacitance in terms of picofarads (pF) instead of farads (F):

.. plot::
   :context:
   :include-source:

   # <previous imports>
   from pmrf.constraints import Positive
   
   class Capacitor(Capacitor):
       C: prf.Param = prf.param(constraint=Positive(), as_free=True, scale=1e-12)
   
       # def a(self, freq: prf.Frequency) -> jnp.ndarray:
           # <same as before>

By passing ``as_free=True``, ParamRF will enforce that the incoming value is a tunable parameter even if a float is passed. Similarly, ``as_fixed=True`` can be used to fix any incoming parameters. However, these converters are entirely optionaly, and by default the parameter's "tunability" is left unchanged, which is the most common use-case (simply registering the value in the parameter hierarchy).

Note that constraints will also always be enforced (even for unconstrained optimizers!), and will also automatically be intersected with any new constraints provided by the caller.

Evaluating the S-parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can now create a capacitor and plot its S-parameters:

.. plot::
   :context:
   :include-source:

   cap = Capacitor(1.0)
   cap.plot_s_db(prf.Frequency(10, 100, 101, 'MHz'), m=1, n=0)

ParamRF will automatically perform the conversion between ABCD and S-parameters internally.
