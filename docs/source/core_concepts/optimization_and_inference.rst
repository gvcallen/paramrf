Optimization and Inference
--------------------------

Models can easily be fit using to measured data using the high-level :func:`pmrf.fit` function, or directly optimized (e.g. for design) using :mod:`pmrf.optimize.minimize` or :mod:`pmrf.infer.sample`.

The lower-level routines directly take an :class:`~pmrf.Evaluator` as one of their arguments to define the objective or log-likelihood function. For design purposes, the built-in :class:`~pmrf.evaluators.Goal` evaluator is useful. For fitting, on the other, data can be passed directly in the format of a :class:`skrf.Network` or :class:`pmrf.NetworkCollection`, features to fit can be defined using strings (e.g. ``'s11'``), and loss or likelihood functions or objects can be passed (such as :class:`pmrf.losses.RMSELoss` or :class:`pmrf.likelihoods.GaussianLikelihood`).


Solvers
^^^^^^^

All of the above methods take a "solver" argument. ParamRF allows for optimization using either :func:`scipy.optimize.minimize` or :func:`optimistix.minimise`, and Bayesian inference using :func:`inferix` (which provides wrappers for **PolyChord** and experimentally for **BlackJAX**).

* **Scipy**: Provides a wrapper around gradient-based and gradient-free optimization algorithms from :func:`scipy.optimize` in :class:`pmrf.optimize.ScipyMinimizer`. This includes algorithms such as *SLSQP*, *Nelder-Mead* and *L-BFGS*. These algorithms are CPU-native and cannot run on the GPU.
* **Optimistix**: Provides JAX-native optimization algorithms, such as :class:`optimistix.BFGS` and :class:`optimistix.NelderMead`. These algorithms run their loop directly in JAX, and therefore can be compiled to any architecture (CPU, GPU, TPU).
* **Inferix**: Enables Bayesian inference through nested sampling and MCMC sampling using e.g. :class:`inferix.PolyChord` and :class:`inferix.NUTS`. This approach provides maximum likelihood parameters, as well as full posterior probability distributions and Bayesian evidence for model comparison. We recommend `this <https://handley-lab.co.uk/nested-sampling-book/intro.html>`_ source for a brief introduction to nested sampling and Bayesian inference.