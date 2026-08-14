Fitting Shared Parameters with a Module
=======================================

:class:`pmrf.Module` is useful when a parameter set has several RF
representations, but is not itself an RF model. In this example, one module holds
a resistor, inductor, and capacitor. Two methods arrange those same components in
different orders, so both measurements constrain one common set of parameters.

Defining the Module and Predictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module has no ``s`` method. Its ``rlc`` and ``clr`` methods explicitly return
RF models, while the predictor selects the two responses used by the fit.

.. plot::
   :context: reset
   :include-source:

   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   import pmrf as prf
   from pmrf.models import Capacitor, Inductor, Resistor

   class SharedRLC(prf.Module):
       resistor: Resistor
       inductor: Inductor
       capacitor: Capacitor

       def rlc(self) -> prf.Model:
           return self.resistor ** self.inductor ** self.capacitor

       def clr(self) -> prf.Model:
           return self.capacitor ** self.inductor ** self.resistor

   def predictor(module, frequency):
       responses = (
           module.rlc().s(frequency)[:, 1, 0],
           module.clr().s(frequency)[:, 1, 0],
       )
       # The optimizer sees one real-valued array containing both complex traces.
       return jnp.stack([
           jnp.concatenate((response.real, response.imag))
           for response in responses
       ])

   frequency = prf.Frequency(0.1, 1.0, 61, "GHz")
   truth = SharedRLC(
       resistor=Resistor(8.0),
       inductor=Inductor(12e-9),
       capacitor=Capacitor(3e-12),
   )
   initial = SharedRLC(
       resistor=Resistor(prf.Bounded(1.0, 20.0, value=10.0)),
       inductor=Inductor(prf.Bounded(5.0, 20.0, value=10.0, scale=1e-9)),
       capacitor=Capacitor(prf.Bounded(1.0, 6.0, value=2.5, scale=1e-12)),
   )

   target = predictor(truth, frequency)
   result = prf.fitting.fit_minimize(
       initial,
       target,
       frequency=frequency,
       features=predictor,
       max_iter=200,
   )

Plotting Both Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fitted result remains a :class:`pmrf.Module`. Its component parameters are
shared automatically because each RF representation is constructed from the same
module fields.

.. plot::
   :context: close-figs
   :include-source:

   def s21_db(model):
       return 20 * jnp.log10(jnp.abs(model.s(frequency)[:, 1, 0]))

   fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
   arrangements = (
       ("R-L-C", lambda module: module.rlc()),
       ("C-L-R", lambda module: module.clr()),
   )

   for axis, (title, network) in zip(axes, arrangements):
       axis.plot(frequency.f_scaled, s21_db(network(truth)), label="Target")
       axis.plot(
           frequency.f_scaled,
           s21_db(network(initial)),
           ":",
           label="Initial",
       )
       axis.plot(
           frequency.f_scaled,
           s21_db(network(result.model)),
           "--",
           label="Fitted",
       )
       axis.set_title(title)
       axis.set_xlabel(f"Frequency ({frequency.unit})")
       axis.grid(True)

   axes[0].set_ylabel("S21 (dB)")
   axes[0].legend()
   fig.tight_layout()

The same pattern extends to joint fits of different fixtures, reference planes,
or circuit topologies: keep the common physical parameters in a module and make
the fitted predictor select the relevant RF models.
