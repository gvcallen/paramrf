JAX Overview
============

ParamRF is built on top of `JAX <https://github.com/google/jax>`_, a high-performance numerical computing library. If you are familiar with standard scientific Python, JAX will feel very natural: its core array object, :class:`jnp.ndarray`, shares an almost identical API to :class:`numpy.ndarray`. However, JAX arrays have additional transformation capabilities and restrictions.

Understanding a few core JAX concepts may be helpful when building or debugging custom models in ParamRF. This section provides a high-level overview of Just-In-Time (JIT) compilation, the XLA graph, autodifferentiation, and vectorization.

JIT Compilation and XLA
-----------------------
Standard Python is interpreted, which can introduce significant overhead. JAX bypasses this by compiling Python functions into optimized machine code using **Just-In-Time (JIT)** compilation, accessed via :func:`jax.jit`.

When a JIT-compiled function is called, it is not immediately executed. Instead, JAX "traces" the function's execution using abstract values. This trace builds a static computation graph in an intermediate representation called **XLA** (Accelerated Linear Algebra). The XLA compiler then optimizes this graph by merging operations, reducing memory allocation, and compiling it to a specific hardware architecture (CPU, GPU, or TPU).

In ParamRF, modules, models, and evaluators are designed to be JIT-compatible. Because JAX relies on static tracing, functions passed to the JIT compiler must be pure. This is why ParamRF modules are immutable; use methods such as :meth:`pmrf.Module.at` instead of modifying attributes in place.

Differentiability and Autodifferentiation
-----------------------------------------
One of the benefits of leveraging JAX is its advanced **autodifferentiation** (autodiff) capabilities. Unlike numerical differentiation (which is slow and prone to truncation/rounding errors) or symbolic differentiation (which scales poorly and leads to expression swell), autodiff sequentially applies the chain rule to the operations in the XLA computation graph. Using transformations like :func:`jax.grad` (for gradients of scalar-valued functions) or :func:`jax.jacfwd` and :func:`jax.jacrev` (for Jacobians of vector-valued functions), JAX computes exact mathematical derivatives:

.. code-block:: python

  import jax
  import jax.numpy as jnp

  def objective(x):
      return jnp.sum(x ** 2)

  grad_fn = jax.grad(objective)
  gradient = grad_fn(jnp.array([1.0, 2.0, 3.0]))  # Returns [2.0, 4.0, 6.0]

In ParamRF, this means you can take gradients with respect to any parameter or frequency. This improves optimization efficiency and stability, since the gradient is immediately known at each iteration.

Vectorization with vmap
-----------------------
:func:`jax.vmap` provides automatic vectorization (or batching) of functions. If you write a function designed to operate on a single set of inputs, wrapping it in :func:`jax.vmap` automatically transforms it to operate over arrays of inputs without the need for Python ``for`` loops.

Equinox and Batched Models
--------------------------
While :func:`jax.vmap` and :func:`jax.jit` are incredibly powerful, they are designed to operate exclusively on JAX arrays. ParamRF models, however, are complex "PyTrees" containing a mix of arrays (parameters) and non-arrays (metadata, strings, or Python booleans). Standard JAX transformations will raise errors when they encounter these non-array elements.

To solve this, ParamRF relies on `Equinox <https://docs.kidger.site/equinox/>`_, which provides "filtered" versions of standard JAX transformations. Functions like :func:`eqx.filter_vmap` and :func:`eqx.filter_jit` inspect the PyTree, safely pass the non-arrays through unchanged, and only apply the transformation to the JAX arrays. 

This allows you to easily create and evaluate entire batches of models simultaneously. To demonstrate this, we can sample a batch of random capacitors. Note that JAX handles randomness differently than standard Python; it requires an explicit pseudo-random number generator (PRNG) state, called a **key** (created via :func:`jax.random.key`), which is passed to random functions and guarantees reproducibility.

.. code-block:: python

  import jax
  import equinox as eqx
  import pmrf as prf
  from pmrf.models import Capacitor
  
  key = jax.random.key(0)
  C_values = jax.random.uniform(key, shape=(10,), minval=1.0e-12, maxval=5.0e-12)
  batched_capacitors = eqx.filter_vmap(Capacitor)(C_values)
  
  freq = prf.Frequency(100, 1000, 100, 'MHz')
  @eqx.filter_jit
  @eqx.filter_vmap
  def evaluate_batch(model):
      return model.s(freq)
  
  s_params = evaluate_batch(batched_capacitors)
  s_params.shape  # (10, 100, 2, 2)

Parax and Unwrapping
--------------------
ParamRF builds on top of the JAX library `Parax <https://github.com/gvcallen/parax>`_ for parameters and constraints. Parax allows parameters (and entire modules) to be manipulated in powerful ways, such as being fixed, scaled, constrained, or tied together. To accomplish this, Parax makes use of a concept known as *unwrapping*. To initialize a *wrapper*, the relevant parameter or object is *wrapped* in the desired class (for example, a "scale" wrapper). Then, to apply the wrapper, the object is *unwrapped*. This mechanism is what allows parameters to be tied together using :class:`pmrf.modules.Tied`, or for parameters to remain bounded, *even* when using an unbounded optimization algorithm.

In ParamRF, unwrapping can be done manually using :func:`pmrf.unwrap` (which is an alias to :func:`parax.unwrap`). It is also automatically applied in ParamRF for RF response methods such as :meth:`pmrf.Model.s` via the :func:`pmrf.unwrap_self` annotation, as well as internally during optimization and inference. However, if you are using non-standard methods on your modules and simply want to evaluate them (for example in a Jupyter notebook), then you should manually unwrap using :func:`pmrf.unwrap`, or annotate your method using :func:`pmrf.unwrap_self` where relevant.

For more information, visit the `Parax <https://gvcallen.github.io/parax>`_ documentation.
