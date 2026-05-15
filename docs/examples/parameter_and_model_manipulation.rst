Parameter and Model Manipulation
================================

All models store their parameters internally. Although it is usually easiest to modify these parameters before you construct the model, it is sometimes convenient to manipulate parameters or swap components already inside a model, or place constraints over multiple parameters.

However, since models are *immutable* and cannot reference each other (to align with JAX's requirements), parameters and sub-models cannot be edited directly (e.g., ``model.R = 50`` will fail) and cannot point to the same objects in memory.

Instead, ParamRF exposes two perform methods to manipulate parameters and sub-models, specifically via :meth:`pmrf.Model.tied` and :attr:`pmrf.Model.at`.

Defining the Base Model
~~~~~~~~~~~~~~~~~~~~~~~

Let's first define a custom RLC composite model to manipulate:

.. code-block:: python

  import pmrf as prf
  from pmrf.models import Resistor, Inductor, Capacitor, Short

  class RLC(prf.Model):
      res: Resistor = Resistor(R=100.0)
      ind: Inductor = Inductor(L=prf.Value(2.0, scale=1e-9))
      cap: Capacitor = Capacitor(C=prf.Value(1.0, scale=1e-12))

      def build(self) -> prf.Model:
          return self.res ** self.ind ** self.cap

  rlc = RLC()

Tied Parameters
~~~~~~~~~~~~~~~

Sub-models and parameters can be *tied together* using :meth:`pmrf.Model.tied`, which returns a new :class:`~pmrf.models.composite.wrapped.Tied` model. For example, to set the resistor to be 100e12 times the capacitor value, we could do the following:

.. code-block:: python

  rlc_tied = rlc.tied(
      lambda m: m.res.R,
      lambda m: m.cap.C,
      lambda c: c*100e12
  )


Modifying Specific Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using :meth:`pmrf.Model.at`, we can traverse down the model tree to focus on a specific parameters and sub-models. For example, we can update values using ``.set()``. This returns a new model with the specified change applied:

.. code-block:: python

  rlc_R200 = rlc.at.res.R.set(200.0)

We can also use this to change the type of the parameter and filter based on condition:

.. code-block:: python

  rlc_fixed = rlc.at.cap.C.set(prf.Fixed(rlc.cap.C))
  rlc_updated = rlc.at.where(lambda x: isinstance(x, Inductor)).L.set(5.0e-9)

Lastly, since a model's structure is not rigid, we can completely swap out components:

.. code-block:: python

  rlc_shorted = rlc.at.ind.set(Short())