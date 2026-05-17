Model Optimization
==================

ParamRF allows you to easily optimize model parameters to meet a given design goal. In this example, we design a simple low-pass filter using SciPy's L-BFGS-B algorithm.

Defining the Model
~~~~~~~~~~~~~~~~~~~

Instead of passing unconstrained floats, we can pass :class:`~pmrf.parameters.Bounded` parameters to specify interval constraints on our values:

.. plot::
   :context: reset
   :include-source:

   import pmrf as prf
   from pmrf.models import ShuntCapacitor, Inductor
   
   C1 = ShuntCapacitor(C=prf.Bounded(1.0, 100.0, scale=1e-12))
   L1 = Inductor(L=prf.Bounded(1.0, 100.0, scale=1e-9))
   C2 = ShuntCapacitor(C=prf.Bounded(1.0, 100.0, scale=1e-12))
   
   lpf = C1 ** L1 ** C2

It is best practice to apply a scaling factor to our values in order to keep the optimization numerically stable. To create fixed parameters and apply more complicated constraints, parameters re-exported from :mod:`pmrf.parameters` can be used e.g. :func:`pmrf.Fixed` and :func:`pmrf.Constrained`.

Running the Optimizer
~~~~~~~~~~~~~~~~~~~~~

Next, we define our design goals. In this case, we want to ensure good matching (low reflection) across our passband. We can use the :class:`~pmrf.evaluators.Goal` evaluator and pass it to the :func:`~pmrf.optimize.minimize` function alongside our frequency range:

.. plot::
   :context:
   :include-source:

   from pmrf.evaluators import Goal
   from pmrf.optimize import minimize, ScipyMinimize

   match_goal = Goal('s11_db', '<', -20)
   passband = prf.Frequency(100, 500, 101, 'MHz')
   result = minimize(match_goal, lpf, passband, solver=ScipyMinimize(method='L-BFGS-B'))

The :func:`~pmrf.optimize.minimize` function returns an :class:`~pmrf.optimize.OptimizeResult` object containing the optimized model and solver metrics. We can extract this newly fitted model to verify our results and print its parameters:

.. plot::
   :context:
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np

   optimized_lpf = result.model

   print("Optimized Parameters:")
   print(optimized_lpf.named_params())

   plt.plot(passband.f, -20.0 * np.ones_like(passband.f), color='black', linestyle='--', label='target')
   lpf.plot_s_db(passband, m=0, n=0, label='initial')
   optimized_lpf.plot_s_db(passband, m=0, n=0, label='optimized')
   plt.title('Initial vs. Optimized S11')

For more complex designs, the :func:`~pmrf.optimize.minimize` function can accept a list of multiple goals, and you can apply masks to evaluate different features across different frequency bands. Custom loss functions can also be specified in :mod:`~pmrf.losses`.

For even more complicated designs, :class:`~pmrf.evaluators.AbstractEvaluator` can be overridden directly.

Note that ParamRF also provides convenience functions for fitting models directly to data in :func:`~pmrf.fitting`. See the tutorial for a detailed guide.