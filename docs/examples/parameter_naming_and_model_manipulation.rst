Parameter Naming and Model Manipulation
=========================================

All models store their parameters internally. Although it is usually easiest to modify these parameters before you construct the model, it is sometimes convenient to manipulate parameters or swap components already inside a model, or place constraints over multiple parameters.

Since models are *immutable* and cannot reference each other (to align with JAX's requirements), parameters and sub-models cannot be edited directly (e.g., ``model.R = 50`` will fail), and also cannot point to the same objects in memory.

Instead, :class:`pmrf.Module` exposes three methods to manipulate parameters and sub-modules: :meth:`pmrf.Module.at`, :meth:`pmrf.Module.tied`, and :meth:`pmrf.Module.map`. These methods accept structural callables or resolved parameter names.

Naming Parameters and Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters and modules can be explicitly named upon construction. ParamRF uses these names to map a module into a flat dictionary via :meth:`pmrf.Module.named_params`. The naming convention resolves dynamically based on your preferences:

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

  class RLC(prf.Module):
      res: Resistor = Resistor(R=100.0, name="myR")
      ind: Inductor = Inductor(L=prf.Unconstrained(2.0, scale=1e-9, name="L_val"), name="myL")
      cap: Capacitor = Capacitor(C=prf.Unconstrained(1.0, scale=1e-12, name="C_global"))

  rlc = RLC()

Modifying Specific Fields
~~~~~~~~~~~~~~~~~~~~~~~~~

Using :meth:`pmrf.Module.at`, we can focus on specific parameters and sub-modules. This returns a lens that allows ``.get()`` or ``.set()`` operations, returning an updated module.

You can pass a single parameter name, an iterable of parameter names, or a functional target:

.. code-block:: python

  # Updating a single value via its string name
  rlc_R200 = rlc.at("myR.R").set(prf.Unconstrained(200.0))
  
  # Updating multiple values simultaneously by passing a tuple of names
  rlc_fixed = rlc.at(("myL_L_val", "C_global")).apply(lambda xs: tuple(prf.as_fixed(x) for x in xs))

Because the model's structure is not rigid, we can also swap entire sub-models. Since sub-models themselves aren't extracted by `.named_params()`, we use functional targets to replace them:

.. code-block:: python

  # Swapping a component out for a Short
  rlc_shorted = rlc.at(lambda m: m.ind).set(Short())

Tied Parameters
~~~~~~~~~~~~~~~

Sub-modules and parameters can be *tied together* using :meth:`pmrf.Module.tied`, which returns a :class:`pmrf.modules.Tied` module. When called on an RF model, ParamRF places that module inside :class:`pmrf.models.Wrapped` so the RF interface remains available. You can define the target and source using string names or callables.

For example, to set the resistor's value to always be 100e12 times the capacitor's value:

.. code-block:: python

  # Using generated parameter names
  rlc_tied = rlc.tied(
      target="myR.R",
      source="C_global",
      tie_fn=lambda c: c * 100e12
  )
  
  # Alternatively, using callables
  rlc_tied_func = rlc.tied(
      target=lambda m: m.res.R,
      source=lambda m: m.cap.C,
      tie_fn=lambda c: c * 100e12
  )

This provides a powerful API for more advanced optimization constraints.
