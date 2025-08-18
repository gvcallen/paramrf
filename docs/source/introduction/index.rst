Introduction
=====================

**ParamRF** provides a declarative circuit modelling interface that compiles circuit models into an efficient linear algebra graph using *JAX*, and also provides the ability to fit or simulate these models using various fitting algorithms and random samplers. An introduction into the core concepts, as well as the showcase of some simple examples, is provided below. A few comparisons are drawn against the popular library *scikit-rf*, which some users may be more familiar with.

Core Concepts
~~~~~~~~~~~~~~~~~~~~

The library revolves around a few key building blocks:

* ``pmrf.Model``: The base class for any computable circuit component, such as foundational models (resistors, transmission lines etc.) or complex circuit models. Compared to scikit-rf's *Network* class, models are *functional* in nature, meaning that they only encapsulate their *representation* (parameters and computation) as opposed to their *data*. Therefore, all model properties such as *s*, *a* etc. accept frequency as an input. Models can be defined using composition such as cascading existing models, or via inheritance of the model class itself.
* ``pmrf.Parameter``: A parameter in a circuit model, storing its value as well as any parameter *metadata*. This allows for parameter bounds and scaling, provides the ability to mark parameters as *fixed*, and can have a prior associated with it for Bayesian fitting.
* ``pmrf.Frequency``: A wrapper around a JAX array that defines the frequency axis over which models are evaluated. The way in which this class is used summarizes the package's main design choice difference compared to *scikit-rf*. Specifically, the frequency object is an *input* to the model, as opposed to a member field of the model itself. This decouples the evaluation of the model from its parameters (and also conveniently provides the ability to differentiate with respect to frequency thanks to *JAX*).

Model Composition
~~~~~~~~~~~~~~~~~~~~
**ParamRF** provides a small component library with commonly-used models, such as lumped and distributed elements, and convenience models that implement functionality such as cascading and port termination. Models can be built directly using these in a compositional approach.

The example below creates an RLC filter and terminates it in an open circuit. The resultant ``rlc`` is a first-class ``Model`` of type ``pmrf.models.containers.Cascade``, consisting of parameters representing the respective *R*, *L* and *C* parameters.

.. code-block:: python

    from pmrf.models import Resistor, Inductor, ShuntCapacitor, OPEN
    from pmrf.parameters import Fixed

    # Instantiate the lumped element models
    resistor = Resistor(R=100.0, name="res")
    inductor = Inductor(L=2e-9)
    capacitor = ShuntCapacitor(C=1e-12)

    # Cascade the models, storing the result, and a terminated version with fixed R
    rlc = resistor ** inductor ** capacitor
    terminated_rlc = rlc.terminated(OPEN).with_params(res_R=Fixed(90.0))

Model Declaration
~~~~~~~~~~~~~~~~~~~~
For more complex or custom circuit or equation-based models, users can inherit directly from ``Model``, which itself is an *Equinox* ``Module`` and therefore a Python ``dataclass`` and *JAX* *PyTree*. Although an in-depth understanding of these concepts is not required, the reader is encouraged to briefly study the *JAX*, *Equinox* and Python documentations for a better overview.

Any attributes of a model are classified as either *static* or *dynamic*. By default, fields of built-in types such as ``str``, ``int``, ``list`` etc. are seen as static in the model hierarchy, whereas those annotated as a ``Parameter`` or ``Model`` are dynamic and can be adjusted (for example, by fitting routines). Parameter initialization is flexible: parameters may be populated with a simple float value; using factory methods such as ``Uniform``, ``Normal`` or ``Fixed``; or directly using the ``Parameter`` class constructor.

The following example demonstrates these concepts by defining an amplifier model represented by some two-port gain terminated in a non-ideal resistor.

.. code-block:: python

    import jax.numpy as jnp
    import pmrf as prf
    from pmrf.models import Resistor, PiCLC, SModel
    from pmrf.parameters import Uniform, Fixed

    # Define a model class. Behaviour is defined by implementing 
    # a primary matrix function such as "s" in our case.
    class TerminatedAmplifier(prf.Model):
        G: prf.Parameter =            Uniform(prf.db_2_mag(10), prf.db_2_mag(15))
        resistor: prf.Model =         Resistor(Fixed(1.0, scale=1e3))
        parasitics: prf.Model =       PiCLC(C1=0.05e-12, L=0.1e-9, C2=0.1e-12)

        def s(self, freq: prf.Frequency):       
            # We use jnp for calculations as a function of freq.f or freq.w,
            # and here wrap it in the utility "SModel" for easier cascading.
            # Note that "terminated()" defaults to a SHORT
            s21 = jnp.sqrt(self.G) * jnp.ones_like(freq.f)
            zeros = jnp.zeros_like(freq.f)
            amp = SModel(jnp.array([
                [zeros, zeros],
                [s21,   zeros]
            ]).transpose(2, 0, 1))
            return (amp ** self.parasitics ** self.res).terminated().s(freq)    

Fitting
~~~~~~~~~~~~~~~~~~~~

A primary application of ``pmrf`` is the fitting of models and their parameters to measured data. The ``pmrf.fitting`` module provides a unified interface to perform this task using either traditional *frequentist* optimization, or *Bayesian* inference techniques.

The general workflow consists of defining a model, loading data via *scikit-rf*, and configuring and running the fitter with the specified settings.

Main Fitters
^^^^^^^^^^^^^^^^^^^^

* ``ScipyMinimizeFitter``: Provides access to gradient-based and gradient-free optimization algorithms from ``scipy.optimize``. This includes algorithms such as *SLSQP*, *Nelder-Mead* and *L-BFGS*.
* ``PolychordFitter``: Enables Bayesian inference through nested sampling. This approach provides maximum likelihood parameter values, as well as full posterior probability distributions and Bayesian evidence useful for model comparison and uncertainty quantification.

Fitting Example
^^^^^^^^^^^^^^^^^^^^

The following provides a complete example of fitting the built in ``PhysicalCoaxial`` model to the measurement of 10m coaxial cable (provided as an example in the `GitHub <https://github.com/paramrf/paramrf/tree/main/examples>`_). Data is loaded using scikit-rf; the model is instantiated with appropriate initial parameters; the fitter is configured with a custom cost function and subsequently run; and results are plotted.

.. code-block:: python

    import jax.numpy as jnp
    import skrf as rf

    import pmrf as prf
    from pmrf.models import PhysicalCoaxial
    from pmrf.parameters import Uniform, Fixed, PercentNormal
    from pmrf.fitting import ScipyMinimizeFitter

    # Load the measured data and setup the model
    measured = rf.Network('data/10m_cable.s2p', f_unit='MHz')
    model = PhysicalCoaxial(
        din = PercentNormal(1.12, 5.0, scale=1e-3),
        dout = PercentNormal(3.2, 5.0, scale=1e-3),
        epr = PercentNormal(1.45, 5.0, n=2),
        rho = PercentNormal(1.6, 5.0, scale=1e-8),
        tand = Uniform(0.0, 0.01, value=0.0, scale=0.01, n=2),
        mur = Fixed(1.0),
        length = PercentNormal(10.0, 5.0),
        epr_model='bpoly',
        tand_model='bpoly',
    )

    # Initialize the fitter. We fit on the real and imaginary "features",
    # and combine their results using a custom cost function
    fitter = ScipyMinimizeFitter(
        model=model,
        measured=measured,
        features=['s11_re', 's11_im'],
        cost=[prf.l2_norm_ax0, jnp.mean, prf.mag_2_db],
    )

    # Run the fit. Arguments are passed through to the underlying solver
    results = fitter.run(method='Nelder-Mead')

    # Convert the model to an skrf Network and plot the resultant S11
    model_ntwk = results.model.to_skrf(measured.frequency)
    model_ntwk.plot_s_db(m=0, n=0, ax=axes[0])
    measured.plot_s_db(m=0, n=0, ax=axes[0])

Sampling
~~~~~~~~~~~~~~~~~~~~

A secondary application of ``pmrf`` is the random sampling or simulation of models using e.g. Latin Hypercube sampling.

The below example demonstrates a very simple example of simulating 10 different resistor networks with uniform resistance between 9 and 11 ohms.

.. code-block:: python

    import pmrf as prf
    from pmrf.models import Resistor
    from pmrf.parameters import Uniform
    from pmrf.sampling import LatinHypercubeSampler

    resistor = Resistor(R=Uniform(9.0, 11.0))
    sampler = LatinHypercubeSampler(resistor)
    resistors = sampler.generate_models(10)
    freq = prf.Frequency(10, 20, 100, 'MHz')

    for i, res in enumerate(resistors):
        res.export_touchstone(freq, f'resistors_{i}')