Installation
=====================
ParamRF can be installed directly using pip:

``pip install paramrf``

Sometimes this version may not be the latest release. To install from GitHub instead:

``pip install git+https://github.com/gvcallen/paramrf@main``

Optional dependencies
---------------------
Several additional dependencies are required/recommended for more advanced use-cases.

For PolyChord fitting:

``pip install git+https://github.com/PolyChord/PolyChordLite.git anesthetic mpi4py``

For BlackJAX fitting:

``pip install git+https://github.com/handley-lab/blackjax@nested_sampling anesthetic``

For eqx-learn surrogate modeling:

``pip install git+https://github.com/eqx-learn/eqx-learn``