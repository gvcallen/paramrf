ParamRF: Parametric Frequency-Domain Modelling, Fitting and Sampling
=====================
**ParamRF**, or ``pmrf``, is an open-source radio frequency (RF) modelling framework. It provides an object-orientated means for frequency-domain modelling, fitting and sampling of RF models, with focus on circuit models. This documentation serves as an introduction into the framework and its features with some basic examples, and also provides an overall API reference.


+----------+------------------------------------------------+
| Version  | |release|                                      |
+----------+------------------------------------------------+
| Author   | Gary Allen                                     |
+----------+------------------------------------------------+
| Homepage | https://github.com/paramrf/paramrf             |
+----------+------------------------------------------------+
| Docs     | https://paramrf.github.io/paramrf              |
+----------+------------------------------------------------+
| Paper    | https://doi.org/10.48550/arXiv.2510.15881      |
+----------+------------------------------------------------+

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   installation
   introduction/index
   api/index
   license


Key Features
---------------------

* **Declarative and Composable Modelling**: Allows for the definition of models using either a self-documenting, declarative syntax or via compositional techniques such as cascading. Since models can consist of a mix of ``Parameter`` objects as well as other ``Model``'s, this allows for a natural means of building complex, hierarchial models from both equations and other sub-models.
* **Unified Fitting Engine**: Provides a number of commonly available fitting algorithms with a unified interface, catering for both classical frequentist optimization and statistical Bayesian inference.
* **JAX Backend**: Leverages `JAX` for Just-In-Time (JIT) compilation of models to high-performance hardware (CPU, GPU, TPU). This removes python overhead due to interpreter context switching; enables better vectorization and parallelization; and provides automatic differentiation through the entire model structure, enabling new analysis and more efficient gradient-based optimization.
* **Extensibility**: Designed to be extendable, such that additional models, fitting algorithms, cost functions, sampling routines etc. can easily be implemented.
* **scikit-rf Integration**: Designed for seamless interoperability with *scikit-rf*, ``pmrf`` models can be evaluated and converted to ``skrf.Network`` objects, providing access to *scikit-rf*'s library of analysis and plotting tools.


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