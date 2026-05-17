Cascading and Terminating
==============================

For simple circuits, models can be built "compositionally" by combining multiple components together.

Using Operators
~~~~~~~~~~~~~~~

A simple example of compositional modeling is cascaded (series) elements. In ParamRF, this can be done using the ``**`` operator for a chain of 2N-port networks:

.. code-block:: python

  import pmrf as prf
  from pmrf.models import ShuntResistor, ShuntInductor, ShuntCapacitor
  
  res, ind, cap = ShuntResistor(100.0), ShuntInductor(2.0e-9), ShuntCapacitor(1.0e-12)
  rlc = res ** ind ** cap

Note that, in the above example, *no computation* was done. Models are *lazy*, and are only evaluated when we pass them a frequency. To evaluate the model, we could extract its S-parameter matrix using :meth:`pmrf.Model.s`:

.. code-block:: python

  freq = prf.Frequency(10, 1000, 101, 'MHz')
  smatrix = rlc.s(freq) # shape (nports, nports, nfreq)
  s11 = smatrix[0,0,:]

ParamRF compiles :meth:`pmrf.Model.s` just-in-time (JIT), and evaluates the batch of frequencies. We can also use the same ``**`` operator for terminating any 2N port in an N port:

.. code-block:: python

  from pmrf.models import Open
  
  open_model = Open()
  rlc_terminated = rlc ** open_model

For a list of built-in models and components, check out the :mod:`pmrf.models` module.

Using Classes
~~~~~~~~~~~~~

The above approach is purely syntactic sugar. We could have achieved the exact same results above by explicitly constructing :class:`~pmrf.models.composite.interconnected.Cascade` and :class:`~pmrf.models.composite.interconnected.Terminated` models:

.. code-block:: python

  from pmrf.models import Cascade, Terminated

  explicit_rlc = Cascade([res, ind, cap])
  explicit_rlc_terminated = Terminated(explicit_rlc, open_model)

This emphasizes that models are purely *containers*, wrapping *parameters*, *other models*, or *static metadata*. This enables arbitrary nesting, providing a powerful foundation for modular modeling and design.