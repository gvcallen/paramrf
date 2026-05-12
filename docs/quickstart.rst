Quickstart
==========

**ParamRF** provides a declarative interface for creating RF circuit and surrogate models in `JAX <https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>`_. This guide provides a brief introduction into defining models, and demonstrates optimization using a simple design goal.

Installation
------------
First, ensure ParamRF is installed (requires Python 3.11+):

.. code-block:: bash

   $ pip install paramrf

Note that a few features (such as statistical modeling) require some additional dependencies. Checkout the homepage for more information.

Creating an RLC model
---------------------
Models can easily be built using composition with the built in :mod:`pmrf.models` library. For example, the ``**`` operator can be used to cascade several models together. In the example below, we define a frequency band, cascade a resistor, inductor, and capacitor to create a series RLC model, and then plot the resultant S-parameters.

.. code-block:: python

  import pmrf as prf
  from pmrf.parameters import Scaled
  from pmrf.models import Resistor, Inductor, Capacitor
  
  # Define the frequency band
  freq = prf.Frequency(1, 10, 101, 'GHz')
  
  # Cascade elements using the ** operator
  rlc_model = Resistor(50) ** Inductor(Scaled(1.0, 1e-9)) ** Capacitor(Scaled(1.0, 1e-12))
  
  # Plot the model's S11 parameter
  rlc_model.plot_s_db(freq, m=0, n=0)

Optimizing the S-parameters
---------------------------
ParamRF provides several optimization and inference wrappers around backends like :mod:`scipy.optimize`, :mod:`Optimistix` and :mod:`PolyChord`. The following snippet demonstrates how to optimize the previous RLC model to satisfy a simple design goal using the built-in :class:`~pmrf.evaluators.Goal` evaluator and the :mod:`scipy.optimize.minimize` backend.

.. code-block:: python

  # Define the optimization frequency and goal
  opt_freq = prf.Frequency(4.0, 6.0, 101, 'GHz')
  goal = prf.evaluators.Goal('s11_db', '<', -20)
  
  # Optimize the previous RLC model
  result = prf.optimize.minimize(goal, rlc_model, opt_freq, solver=prf.optimize.ScipyMinimize())
  
  # Plot the optimized model
  result.model.plot_s_db(freq, m=0, n=0)


Next steps
----------
* To delve a bit deeper into understanding the library's core building blocks, see the :doc:`core_concepts/index` page.
* For a step-by-step guide, the :doc:`tutorials/index` is a good place to start.