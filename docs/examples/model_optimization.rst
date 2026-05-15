Model Optimization
==================

ParamRF allows you to easily optimize model parameters to meet a given design goal. In this example, we design a simple low-pass filter.

Defining the Model
~~~~~~~~~~~~~~~~~~~

Instead of passing unconstrained floats, we can pass :class:`~pmrf.parameters.Bounded` parameters to specify constraints on our values:

.. code-block:: python

  import pmrf as prf
  from pmrf.models import ShuntCapacitor, Inductor
  
  C1 = ShuntCapacitor(C=prf.Bounded(1.0, 100.0, scale=1e-12))
  L1 = Inductor(L=prf.Bounded(1.0, 100.0, scale=1e-9))
  C2 = ShuntCapacitor(C=prf.Bounded(1.0, 100.0, scale=1e-12))
  
  lpf = C1 ** L1 ** C2

It is best practice to apply a scaling factor to our values, in order to keep the optimization numerically stable. To create fixed parameters and apply more complicated constraints, see :func:`pmrf.Fixed` and :func:`pmrf.Constrained`.

Running the Optimizer
~~~~~~~~~~~~~~~~~~~~~

Next, we define our design goals. In this case, we want to ensure good matching (low reflection) across our passband. We can use the :class:`~pmrf.evaluators.Goal` evaluator and pass it to the :func:`~pmrf.optimize.minimize` function alongside our frequency range:

.. code-block:: python

  from pmrf.evaluators import Goal

  match_goal = Goal('s11_db', '<', -20)
  passband = prf.Frequency(100, 500, 101, 'MHz')
  result = prf.optimize.minimize(match_goal, lpf, passband)

The :func:`~pmrf.optimize.minimize` function returns an :class:`~pmrf.optimize.OptimizeResult` object containing the optimized model and solver metrics. We can extract this newly fitted model to verify our results and print its parameters:

.. code-block:: python

  optimized_lpf = result.model
  optimized_lpf.plot_s_db(passband, m=0, n=0)

  print(optimized_lpf.named_params())

For more complex designs, the :func:`~pmrf.optimize.minimize` function can accept a list of multiple goals, and you can apply masks to evaluate different features across different frequency bands. Custom loss functions can also be specified in :mod:`~pmrf.losses`.

For even more complicated designs, :class:`~pmrf.evaluators.AbstractEvaluator` can be overridden directly.

Note that ParamRF also provides convenience functions for fitting models directly to data in :func:`~pmrf.fitting`. See the tutorial for a detailed guide.