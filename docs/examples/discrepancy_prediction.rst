Discrepancy Prediction
======================

A fit with a Gaussian-process discrepancy learns how the model differs from the data, not just the model parameters. That discrepancy can be predicted at frequencies the fit never saw, with its uncertainty. In this example we fit the S11 of a 10 m coaxial line with a GP discrepancy, then predict the discrepancy off the fit grid.

Fitting with a Discrepancy
~~~~~~~~~~~~~~~~~~~~~~~~~~

The measured S11 is a coaxial line plus a slow ripple that the model cannot produce, and a little measurement noise. We fit the dielectric constant, with a :class:`~pmrf.discrepancy_models.GaussianProcess` to absorb the ripple. Its kernel's length scale is in the fit frequency's unit, here MHz:

.. plot::
   :context: reset
   :include-source:
   :nofigs:

   import numpy as np
   import pmrf as prf
   from pmrf.models import CoaxialLine
   from pmrf.materials import BulkConductor, ConstantDielectric
   from pmrf.fitting import fit
   from pmrf.likelihoods import GaussianLikelihood
   from pmrf.discrepancy_models import GaussianProcess
   from pmrf.covariance_kernels import Matern52Kernel

   def coax(ep_r):
       return CoaxialLine(
           d_in=1.05e-3, d_out=3.2e-3,
           dielectric=ConstantDielectric(ep_r=ep_r, tand=1e-3),
           conductor=BulkConductor(sigma=1 / 1.7e-8),
           length=10.0,
       )

   freq = prf.Frequency(10, 500, 100, 'MHz')
   ripple = 0.01 * np.exp(2j * np.pi * freq.f_scaled / 150.0)
   rng = np.random.default_rng(0)
   noise = 1e-3 * (rng.normal(size=freq.npoints) + 1j * rng.normal(size=freq.npoints))
   s11 = np.asarray(coax(1.40).s(freq)[:, 0, 0]) + ripple + noise

   gp = GaussianProcess(
       Matern52Kernel(prf.Bounded(10.0, 200.0, value=50.0)) * prf.Bounded(1e-5, 1e-3, value=1e-4)
   )
   result = fit(
       coax(prf.Bounded(1.38, 1.42, value=1.39)), s11, frequency=freq, features='s11',
       likelihood=GaussianLikelihood(prf.Bounded(1e-7, 1e-4, value=1e-6)),
       discrepancy=gp,
   )

Predicting the Discrepancy
~~~~~~~~~~~~~~~~~~~~~~~~~~

The fitted :class:`~pmrf.evaluators.MarginalLogLikelihood`, with its optimized GP hyperparameters and noise, is the one term of the fit's objective. :meth:`~pmrf.evaluators.MarginalLogLikelihood.predict_discrepancy` conditions the GP on the fit's residuals and returns the distribution of the discrepancy at new frequencies. Here they extend past the fit band and are given in GHz; they are converted to the fit's unit first.

.. plot::
   :context:
   :include-source:

   import matplotlib.pyplot as plt

   (term,) = result.solution.objective
   mll = term.evaluator.evaluator

   new_freq = prf.Frequency(0.005, 0.6, 300, 'GHz')
   delta = mll.predict_discrepancy(result.model, freq, new_freq)

   mean = delta.mean()
   std = np.sqrt(np.diagonal(delta.covariance(), axis1=-2, axis2=-1))

   fig, axes = plt.subplots(2, 1, sharex=True)
   for ax, m, s, r, part in zip(axes, mean, std, (ripple.real, ripple.imag), ('Re', 'Im')):
       ax.fill_between(new_freq.f_scaled * 1e3, m - s, m + s, alpha=0.3, label=r'$\pm 1\sigma$')
       ax.plot(new_freq.f_scaled * 1e3, m, label='predicted mean')
       ax.plot(freq.f_scaled, r, 'k.', markersize=3, label='true ripple')
       ax.set_ylabel(rf'{part} $\delta$')
   axes[0].legend()
   axes[1].set_xlabel('Frequency (MHz)')

The prediction is in event space: one block for the real part of S11 and one for the imaginary part, with frequency last. It is of the discrepancy alone, and excludes measurement noise. Inside the fit band it follows the ripple closely; beyond it, the mean relaxes towards zero and the uncertainty grows back to the GP's prior.
