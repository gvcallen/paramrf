JAX and Batched Models
======================

ParamRF is built on top of `JAX <https://github.com/google/jax>`_, a high-performance numerical computing library. If you are familiar with standard scientific Python, JAX will feel very natural: its core array object, :class:`jnp.ndarray`, shares an almost identical API to :class:`numpy.ndarray`. However, JAX extends this familiar interface with powerful transformations.

Understanding a few core JAX concepts may be helpful when building custom components or debugging optimization routines in ParamRF. This section provides a high-level technical overview of Just-In-Time (JIT) compilation, the XLA graph, autodifferentiation, and vectorization.

JIT Compilation and XLA
--------------------------------
Standard Python is interpreted, which introduces significant overhead for heavy numerical workloads. JAX bypasses this by compiling Python functions into optimized machine code using **Just-In-Time (JIT)** compilation, accessed via :func:`jax.jit`.

When a JIT-compiled function is called, it is not immediately executed. Instead, JAX "traces" the function's execution using abstract values to map out its sequence of mathematical operations. This trace builds a static computation graph in an intermediate representation called **XLA** (Accelerated Linear Algebra). The XLA compiler then optimizes this graph by fusing operations, reducing memory allocation, and compiling it to a specific hardware architecture (CPU, GPU, or TPU).

In ParamRF, all model evaluations and optimization loops are designed to be JIT-compatible. Because JAX relies on this static tracing mechanism, functions passed to the JIT compiler must be *pure* (stateless and lacking side-effects). This functional paradigm is the underlying reason why ParamRF models are immutable; rather than modifying a model's attributes in-place, you use functional methods like :meth:`pmrf.Model.at` to generate a new model state.

Differentiability and Autodifferentiation
-----------------------------------------
One of benefits of leveraging JAX is its advanced **autodifferentiation** (autodiff) capabilities. Unlike numerical differentiation (which is slow and prone to truncation/rounding errors) or symbolic differentiation (which scales poorly and leads to expression swell), autodiff systematically applies the chain rule to the sequence of operations in the XLA computation graph.

Using transformations like :func:`jax.grad` (for gradients of scalar-valued functions) or :func:`jax.jacfwd` and :func:`jax.jacrev` (for Jacobians of vector-valued functions), JAX computes exact mathematical derivatives of arbitrarily complex code.

.. code-block:: python

  import jax
  import jax.numpy as jnp

  def objective(x):
      return jnp.sum(x ** 2)

  # Automatically compute the exact gradient of the objective function
  grad_fn = jax.grad(objective)
  gradient = grad_fn(jnp.array([1.0, 2.0, 3.0]))  # Returns [2.0, 4.0, 6.0]

In ParamRF, this means you can build deep, hierarchical models, and take gradients with respect to any parameter and frequency. This not only improves optimization efficiency and stability (since the gradient is immediately known at each iteration), but also may allow for new design opportunities.

Vectorization with vmap
-----------------------
:func:`jax.vmap` provides automatic vectorization (or batching) of functions. If you write a function designed to operate on a single set of inputs, wrapping it in :func:`jax.vmap` automatically transforms it to operate efficiently over arrays of inputs without the need for Python ``for`` loops.

Behind the scenes, JAX pushes the batching dimensions down into the compiled XLA primitive operations, executing them in parallel where hardware permits.

Batched Models and Equinox
--------------------------
While :func:`jax.vmap` and :func:`jax.jit` are incredibly powerful, they are designed to operate exclusively on JAX arrays. ParamRF models, however, are complex "PyTrees" containing a mix of arrays (parameters) and non-arrays (metadata, strings, or Python booleans). Standard JAX transformations will raise errors when they encounter these non-array elements.

To solve this, ParamRF relies on `Equinox <https://docs.kidger.site/equinox/>`_, which provides "filtered" versions of standard JAX transformations. Functions like :func:`eqx.filter_vmap` and :func:`eqx.filter_jit` inspect the PyTree, safely pass the non-arrays through unchanged, and only apply the transformation to the JAX arrays. 

This allows you to easily create and evaluate entire batches of models simultaneously. To demonstrate this, we can sample a batch of random capacitors. Note that JAX handles randomness differently than standard Python; it requires an explicit pseudo-random number generator (PRNG) state, called a **key** (created via :func:`jax.random.key`), which is passed explicitly to random functions to guarantee reproducibility.

.. code-block:: python

  import jax
  import equinox as eqx
  import pmrf as prf
  from pmrf.models import Capacitor
  
  # 1. Create an explicit JAX PRNG key
  key = jax.random.key(42)
  
  # Sample 10 random capacitance values between 1pF and 5pF
  C_values = jax.random.uniform(key, shape=(10,), minval=1.0e-12, maxval=5.0e-12)
  
  # 2. Create a batch of 10 Capacitors
  # We use eqx.filter_vmap to vectorize the model instantiation over the C_values array
  batched_capacitors = eqx.filter_vmap(Capacitor)(C_values)
  
  # 3. Evaluate S-parameters for the whole batch concurrently
  freq = prf.Frequency(100, 1000, 100, 'MHz')
  
  # We filter_jit the function for speed, and filter_vmap to handle the batching.
  @eqx.filter_jit
  @eqx.filter_vmap()
  def evaluate_batch(model):
      return model.s(freq)
  
  # Returns batched S-parameters with shape (10, nfreq, 2, 2)
  s_params = evaluate_batch(batched_capacitors)

Unwrapping and Frozen/Static Modules
------------------------------------
ParamRF builds on top of a JAX library known as `Parax <https://github.com/gvcallen/parax>`_ to allow for parameter constraints, fixing, and other tools to make model optimization and inference easier. To accomplish this, Parax makes use of a concept known as `unwrapping`, which allows applying a "computation" to a JAX PyTree (e.g. constraints, scaling etc.) at a known time. For more information, visit the Parax documentation.