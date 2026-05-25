Parameter Naming and Model Manipulation
=========================================

All models store their parameters internally. Although it is usually easiest to modify these parameters before you construct the model, it is sometimes convenient to manipulate parameters or swap components already inside a model, or place constraints over multiple parameters.

Since models are *immutable* and cannot reference each other (to align with JAX's requirements), parameters and sub-models cannot be edited directly (e.g., ``model.R = 50`` will fail), and also cannot point to the same objects in memory.

Instead, ParamRF exposes three methods to manipulate parameters and sub-models: :meth:`pmrf.Model.at`, :meth:`pmrf.Model.tied`, and :meth:`pmrf.Model.map`. These manipulation methods allow you to traverse your model using either structural callables (lambdas) or resolved string names.

Naming Parameters and Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters and models can be explicitly named upon construction. ParamRF uses these names to map the structure of a model into a flat dictionary via :meth:`pmrf.Model.named_params`. The naming convention resolves dynamically based on your preferences:

1. If no custom names are present, the standard Python attribute path is used (e.g., ``'res.R'``).
2. If models within the structural path have names, they are joined using a separator (default ``_``) to form a namespace prefix. 
3. If the parameter itself is named, it is appended to this namespace prefix, dropping the generic attribute paths.

This approach gives you the flexibility to use a flat naming convention (naming only the parameters), a namespace convention (naming only the models), a fully nested convention, or a combination of these.

Defining the Base Model
~~~~~~~~~~~~~~~~~~~~~~~

Let's define a custom RLC composite model with explicit names to demonstrate how naming and manipulation work together:

.. code-block:: python

  import pmrf as prf
  from pmrf.models import Resistor, Inductor, Capacitor, Short

  class RLC(prf.Model):
      res: Resistor = Resistor(R=100.0, name="myR")
      ind: Inductor = Inductor(L=prf.Unconstrained(2.0, scale=1e-9, name="L_val"), name="myL")
      cap: Capacitor = Capacitor(C=prf.Unconstrained(1.0, scale=1e-12, name="C_global"))

      def build(self) -> prf.Model:
          return self.res ** self.ind ** self.cap

  rlc = RLC()

Modifying Specific Fields
~~~~~~~~~~~~~~~~~~~~~~~~~

Using :meth:`pmrf.Model.at`, we can focus on specific parameters and sub-models. This returns a lens that allows you to cleanly ``.get()`` or ``.set()`` values, returning a new model with the specified change applied. 

You can pass a single parameter name, an iterable of parameter names, or a functional target:

.. code-block:: python

  # Updating a single value via its string name
  rlc_R200 = rlc.at("myR.R").set(prf.Unconstrained(200.0))

  # Updating multiple values simultaneously by passing a tuple of names
  rlc_fixed = rlc.at(("myL_L_val", "C_global")).apply(lambda xs: prf.as_fixed(x) for x in xs)

Because the model's structure is not rigid, we can also swap entire sub-models. Since sub-models themselves aren't extracted by `.named_params()`, we use functional targets to replace them:

.. code-block:: python

  # Swapping a component out for a Short
  rlc_shorted = rlc.at(lambda m: m.ind).set(Short())

Tied Parameters
~~~~~~~~~~~~~~~

Sub-models and parameters can be *tied together* using :meth:`pmrf.Model.tied`, which returns a new :class:`~pmrf.models.composite.wrapped.Tied` model. You can define the target and source using string names or callables. 

For example, to set the resistor's value to always be 100e12 times the capacitor's value:

.. code-block:: python

  # Using generated parameter names
  rlc_tied = rlc.tied(
      target="myR_R",
      source="myC_C_val",
      tie_fn=lambda c: c * 100e12
  )

  # Alternatively, using callables
  rlc_tied_func = rlc.tied(
      target=lambda m: m.res.R,
      source=lambda m: m.cap.C,
      tie_fn=lambda c: c * 100e12
  )

This provides a powerful API for more advanced optimization constraints.