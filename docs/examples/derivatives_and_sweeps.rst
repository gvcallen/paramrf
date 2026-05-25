Derivatives and Sweeps
=========================

ParamRF provides built-in utilities for the calculation of analytical derivatives, as well as for performing vectorized parameter and function sweeps.

Model Setup
~~~~~~~~~~~

First, let's set up a base low-pass filter example model with named models:

.. plot::
   :context: reset
   :include-source:

   from pmrf.models import ShuntCapacitor, Inductor, Cascade
   
   c1 = ShuntCapacitor(C=1.0e-12, name='c1')
   l1 = Inductor(L=1.0e-9, name='l1')
   c2 = ShuntCapacitor(C=1.0e-12, name='c2')
   
   lpf = Cascade([c1, l1, c2])

Scalar Derivatives
~~~~~~~~~~~~~~~~~~


To differentiate a model's parameters, we can pass any differentiable function to :func:`pmrf.derivative`, as well as its nominal arguments:

.. plot::
   :context:
   :include-source:

   import pmrf as prf
   
   freq = prf.Frequency(2.4, 2.4, 1, 'GHz')
   
   def s21_mag(model):
       return model.s_mag(freq)[0,1,0]
   
   (derivatives,) = prf.derivative(s21_mag, lpf)
   
   print(f"Sensitivity to C1: {derivatives.at('c1.C').get() * 1e-12:.3f} / pF")
   print(f"Sensitivity to L: {derivatives.at('l1.L').get() * 1e-9:.3f} / nH")

**Output:**

.. code-block:: none

   Sensitivity to C1: -0.106 / pF
   Sensitivity to L: 0.086 / nH

Note that we had to manually "unwrap" our model before taking gradients in this manner. This is because all parameters in ParamRF are wrappers around raw JAX arrays, and we want to take derivatives with respect to their physical parameter values, and not the raw, underlying optimizer values. (For other tasks in ParamRF, like optimization and inference, this is often done automatically).


Broadcasting Sensitivities Across a Band
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To evaluate sensitivity across a frequency band, JAX's reverse-mode Jacobian function :func:`jax.jacrev` can be used, which computes the derivative of an S-parameter array with respect to specific inputs:

.. plot::
   :context:
   :include-source:

   import jax
   import matplotlib.pyplot as plt
   
   band = prf.Frequency(1, 5, 201, 'GHz')
   
   def s21_mag_array(c1_val, l_val):
       model = ShuntCapacitor(C=c1_val) ** Inductor(L=l_val) ** ShuntCapacitor(C=1.0e-12)
       return model.s_mag(band)[:,1,0]
   
   jacobian_fn = jax.jacrev(s21_mag_array, argnums=(0, 1))
   
   c_nom, l_nom = 1.0e-12, 1.0e-9
   ds21_dc, ds21_dl = jacobian_fn(c_nom, l_nom)
   
We can plot the results as a function of frequency to visualize their behaviour:

.. plot::
   :context:
   :include-source:
      
   fig, ax1 = plt.subplots(figsize=(8, 5))

   ax1.plot(band.f_scaled, ds21_dc * 1e-12, color='tab:blue', label='C1 Sensitivity / pF')
   ax1.set_xlabel('Frequency (GHz)')
   ax1.set_ylabel(r'$\partial |S_{21}| / \partial C$', color='tab:blue')
   ax1.tick_params(axis='y', labelcolor='tab:blue')
   
   ax2 = ax1.twinx()
   ax2.plot(band.f_scaled, ds21_dl * 1e-9, color='tab:red', linestyle='--', label='L Sensitivity / nH')
   ax2.set_ylabel(r'$\partial |S_{21}| / \partial L$', color='tab:red')
   ax2.tick_params(axis='y', labelcolor='tab:red')
   
   plt.title('Sensitivity of $S_{21}$ Magnitude')
   fig.legend()
   fig.tight_layout()