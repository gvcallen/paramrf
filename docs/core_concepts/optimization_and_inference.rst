Optimization and Inference
==========================

ParamRF provides a higher-level interface for model optimization and inference. Models can be optimized using :mod:`pmrf.optimize.minimize`, sampled for statistical inference using :mod:`pmrf.infer.sample`, or fit to measured data using the high-level routines in :mod:`pmrf.fitting`.

JAX vs CPU
~~~~~~~~~~

ParamRF provides two types of solvers, namely *JAX-native*, and *CPU-native* solvers. JAX-native solvers are implemented fully in JAX code, while CPU-native solvers are simply regular solvers wrapped in a JAX callback interface. These solvers act fundamentally differently, and it is important to understand their strengths and weaknesses.

The entire optimization loop for JAX-native solvers can be just-in-time (JIT) compiled. This means that they can run on different platforms (CPU, GPU etc.). However, they are structured slightly differently to standard solvers. For example, JAX code *does not allow any dynamic memory allocation* after compilation. This means that JIT-compiled optimizers, for example, will always run for a *fixed number of iterations* before terminating.

Conversely, CPU-native solvers *only compile the forward pass* (such as the objective function), meaning they can be early-stopped, but *cannot be reused* without recompiling the forward pass. For simple models, this overhead may be negligible, but complicated models likely should prefer JAX-native solutions to avoid model recompilation.

Frequentist vs Bayesian
~~~~~~~~~~~~~~~~~~~~~~~

While "frequentist" optimization provides a single best set of parameters for your model, "Bayesian" inference provides a full probability distribution over your parameters. This can be useful for when you want to explore the full possibility of parameters that satisfy your goal function or fit your data with some probability. ParamRF provides Bayesian inference out-of-the box, and applying it to circuit modeling is an active area of research. We recommend `this source <https://www.stat.cmu.edu/~larry/=sml/Bayes.pdf>`_ for a brief introduction to Bayesian inference and Bayesian sampling approaches.

Availble Solvers
~~~~~~~~~~~~~~~~

ParamRF allows for optimization and inference using a number of built-in backends, namely :func:`scipy`, :func:`optimistix`, :mod:`jaxopt`, :func:`pypolychord` and :mod:`blackjax`. These can be found in :mod:`pmrf.optimize` and :mod:`pmrf.infer`.

* **Optimistix**: JAX-native optimization algorithms, such as :class:`pmrf.optimize.LBFGS` and :class:`pmrf.optimize.NelderMead`.
* **JAXopt**: More JAX-native algorithms, such as :class:`pmrf.optimize.LBFGSB`.
* **Scipy**: A wrapper around CPU-native gradient-based and gradient-free optimization algorithms from :func:`scipy.optimize.minimize` in :class:`pmrf.optimize.ScipyMinimize`. This includes algorithms such as *SLSQP*, *Nelder-Mead* and *L-BFGS*. These are often more robust than their JAX counterparts.
* **BlackJAX**: JAX-native Bayesian inference algorithms, e.g. :class:`pmrf.infer.NUTS` and :class:`pmrf.infer.HMC`.
* **PolyChord**: A wrapper around the CPU-native PolyChord nested sampling algorithm from :func:`pypolychord.run` in :class:`pmrf.infer.PolyChord`.

Custom solvers can also be implemented by overriding the relevant abstract interface, such as :class:`pmrf.optimize.AbstractUnconstrainedMinimizer`, :class:`pmrf.optimize.AbstractBoundedMinimizer`, :class:`pmrf.infer.AbstractJointSampler`, etc.