Quickstart
==========

**ParamRF** provides a declarative interface for creating RF models, such as circuit models, in `JAX <https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>`_. This guide provides a brief introduction to defining and evaluating basic models, as well as fitting them to data.

Installation
------------
First, ensure ParamRF is installed (requires Python 3.11+):

.. code-block:: bash

   $ pip install paramrf

Note that for Bayesian inference or complex statistical modelling, you may need this fork of distreqx:

.. code-block:: bash

   $ pip install git+https://github.com/gvcallen/distreqx.git

Creating an RLC model
---------------------
ParamRF allows models to be built compositionally. For cascaded circuits, the ``**`` operator can be used to chain several models together. The example below defines a frequency band, cascades a resistor, inductor, and capacitor to form an RLC filter, and evaluates its S-parameters.

.. code-block:: python

  import pmrf as prf
  from pmrf.models import Resistor, Inductor, Capacitor
  
  # Define the frequency band
  freq = prf.Frequency(1, 10, 101, 'GHz')
  
  # Cascade elements using the ** operator
  rlc_model = Resistor(50) ** Inductor(1e-9) ** Capacitor(1e-12)
  
  # Plot the model's S11 parameter
  rlc_model.plot_s_db(freq, m=0, n=0)

Optimizing S-parameters
---------------------------
ParamRF provides a several optimization, fitting and sampling wrappers around backends like :mod:`scipy.minimize` and :mod:`Optimistix`. The following snippet demonstrates how to optimize the above RLC model to a simple design goal.

.. code-block:: python

  # Define the optimization frequency and goal
  opt_freq = prf.Frequency(4.0, 6.0, 101, 'GHz')
  goal = prf.evaluators.Goal('s11_db', '<', -20)
  
  # Optimize the previous RLC model. For this problem, Nelder-Mead works well.
  result = prf.optimize.minimize(goal, rlc_model, opt_freq, solver='Nelder-Mead')
  
  # Plot the result at the original frequency
  result.model.plot_s_db(freq, m=0, n=0)


Next steps
----------
* To developer a better understanding of the above code and the library's building blocks (e.g. :class:`pmrf.Model`), see the :doc:`core_concepts/index` page.
* For step-by-step guides on advanced features like model fitting and Bayesian inference, head over to the :doc:`tutorials/index` section.