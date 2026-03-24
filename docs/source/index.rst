ParamRF: Parametric Radio Frequency Modelling, Optimization and Sampling
===================================================================
**ParamRF**, or ``pmrf``, is an open-source radio frequency (RF) modelling framework. It provides a declarative, object-orientated syntax for modelling complex RF circuits and surrogates, as well as their optimization, fitting, statistical analysis and Bayesian inference.

+----------+------------------------------------------------+
| Version  | |release|                                      |
+----------+------------------------------------------------+
| Author   | Gary Allen                                     |
+----------+------------------------------------------------+
| Homepage | https://github.com/gvcallen/paramrf             |
+----------+------------------------------------------------+
| Docs     | https://gvcallen.github.io/paramrf              |
+----------+------------------------------------------------+
| Paper    | https://doi.org/10.48550/arXiv.2510.15881      |
+----------+------------------------------------------------+


Key Features
---------------------

* **Declarative syntax**: Allows for the definition of models using either a self-documenting, declarative syntax, or via compositional techniques such as cascading or node composition. Since models can consist of a mix of ``parax.Parameter`` and other ``pmrf.Model`` objects, this allows for a natural means of building complex, hierarchial models.
* **Automatic differentation**: Since the framework is built using ``jax``, all models can be differentiation with respect to frequency and parameters. This allows for complex optimization and sensitivity analysis.
* **High performance/hardware flexibile**: Since models are compiled using ``jax`` with Just-In-Time (JIT) compilation, model performance is improved and can also be computed on high-performance hardware (CPU, GPU, TPU).
* **Built-in optimization and inference**: Provides built-in wrappers for frequentist optimization and Bayesian inference in ``pmrf.optimize`` and ``pmrf.infer``, as well as high-level wrappers for data-fitting in ``pmrf.fit``.
* **Extensibility**: Designed to be extendable, such that additional models, fitting algorithms, cost functions, sampling routines etc. can easily be implemented.


.. toctree::
   :maxdepth: 2
   :caption: Documentation

   installation
   introduction/index
   api/index
   skrf_comparison
   license

Citation
---------------------

If you have used ParamRF for academic work, please cite the original paper (https://doi.org/10.48550/arXiv.2510.15881):
as: ::

   G.V.C. Allen, D.I.L. de Villiers, (2025). ParamRF: A JAX-native Framework for Declarative Circuit Modelling. arXiv, https://doi.org/10.48550/arXiv.2510.15881.

or using the BibTeX:

.. code:: bibtex

   @article{paramrf,
      doi = {10.48550/arXiv.2510.15881},
      url = {https://doi.org/10.48550/arXiv.2510.15881}, 
      year = {2025},
      month = {Oct},
      title = {ParamRF: A JAX-native Framework for Declarative Circuit Modelling}, 
      author = {Gary V. C. Allen and Dirk I. L. de Villiers},
      eprint = {2510.15881},
      archivePrefix = {arXiv},
      primaryClass = {cs.OH},
   }