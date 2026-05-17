Automatic Differentiation
=========================

ParamRF is built on JAX and Equinox, making models natively differentiable. This allows for analytical gradients of circuit responses with respect to component values, avoiding the need for numerical approximations.

Computing Parameter Sensitivities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To differentiate a model's parameters, define a pure function that returns a scalar metric (e.g., insertion loss). We use ``equinox.filter_grad`` to compute derivatives for the continuous parameters within the model tree.

.. plot::
   :context: reset
   :include-source:

   import pmrf as prf
   from pmrf.models import ShuntCapacitor, Inductor, Cascade
   import jax.numpy as jnp
   import equinox as eqx

   c1 = ShuntCapacitor(C=1.2e-12)
   l1 = Inductor(L=3.3e-9)
   c2 = ShuntCapacitor(C=1.2e-12)
   lpf = Cascade([c1, l1, c2])
   
   freq = prf.Frequency(2.4, 2.4, 1, 'GHz')

   def s21_mag_sq(model):
       s = model.s(freq)
       s21 = s[0, 1, 0]
       return jnp.real(s21 * jnp.conj(s21))

   grad_fn = eqx.filter_grad(s21_mag_sq)
   sensitivities = grad_fn(lpf)

   print(f"Sensitivity to first Capacitor: {sensitivities.models[0].C}")
   print(f"Sensitivity to Inductor: {sensitivities.models[1].L}")

Broadcasting Sensitivities Across a Band
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To evaluate sensitivity across a frequency band, use JAX's reverse-mode Jacobian (``jax.jacrev``). This computes the derivative of an S-parameter array with respect to specific inputs.

.. plot::
   :context:
   :include-source:

   import jax
   import matplotlib.pyplot as plt

   band = prf.Frequency(1, 5, 201, 'GHz')

   def s11_mag_array(c_val, l_val):
       model = ShuntCapacitor(C=c_val) ** Inductor(L=l_val) ** ShuntCapacitor(C=c_val)
       s = model.s(band)
       return jnp.abs(s[:, 0, 0])

   jacobian_fn = jax.jacrev(s11_mag_array, argnums=(0, 1))
   
   c_nom, l_nom = 1.2e-12, 3.3e-9
   ds11_dc, ds11_dl = jacobian_fn(c_nom, l_nom)

   fig, ax1 = plt.subplots(figsize=(8, 5))
   
   ax1.plot(band.f_scaled, ds11_dc, color='tab:blue', label='Sensitivity to C')
   ax1.set_xlabel('Frequency (GHz)')
   ax1.set_ylabel(r'$\partial |S_{11}| / \partial C$', color='tab:blue')
   ax1.tick_params(axis='y', labelcolor='tab:blue')

   ax2 = ax1.twinx()
   ax2.plot(band.f_scaled, ds11_dl, color='tab:red', linestyle='--', label='Sensitivity to L')
   ax2.set_ylabel(r'$\partial |S_{11}| / \partial L$', color='tab:red')
   ax2.tick_params(axis='y', labelcolor='tab:red')

   plt.title('Sensitivity of $S_{11}$ Magnitude')
   fig.tight_layout()