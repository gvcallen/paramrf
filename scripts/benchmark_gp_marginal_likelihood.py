"""Benchmark a Gaussian-process marginal log-likelihood.

A two-port ``DatasheetLine`` is scored against N = 1000 points from 10 to 500 MHz,
with a shared auto/cross GP discrepancy. Each row reports the minimum per-call time
over a few repetitions, after one warm-up call that compiles.

``eqx.filter_jit`` treats Python-float hyperparameters as static, so a gradient with
float hyperparameters never differentiates through the kernel. The float gradient
row is therefore taken with respect to the model only, and the ``Random`` gradient
row with respect to both the evaluator and the model.

Usage: python scripts/benchmark_gp_marginal_likelihood.py [--npoints N] [--repeats R]
"""
import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import pmrf as prf
from pmrf.covariance_kernels import (
    AutoCrossKernel,
    Matern52Kernel,
    PeriodicKernel,
    SharedIndependentKernel,
)
from pmrf.discrepancy_models import GaussianProcess
from pmrf.distributions import RelativeTruncatedNormal
from pmrf.evaluators import MarginalLogLikelihood
from pmrf.likelihoods import GaussianLikelihood
from pmrf.models import DatasheetLine


def build(npoints: int, random: bool):
    """Return ``(mll, model, frequency)`` for the two-port GP benchmark.

    With ``random``, every kernel hyperparameter is a ``prf.Random`` centred on its
    float value; otherwise it is that float.
    """
    def h(value, name):
        if not random:
            return value
        return prf.Random(RelativeTruncatedNormal(value, 0.1), name=name)

    k_auto = (
        PeriodicKernel(h(12.0, 'auto_period'), h(1.5, 'auto_periodic_ls'))
        * Matern52Kernel(h(100.0, 'auto_envelope_ls'))
        * h(2e-5, 'auto_periodic_var')
        + Matern52Kernel(h(4.0, 'auto_ls')) * h(3e-5, 'auto_var')
    )
    k_cross = (
        PeriodicKernel(h(24.0, 'cross_period'), h(1.5, 'cross_periodic_ls'))
        * Matern52Kernel(h(60.0, 'cross_envelope_ls'))
        * h(8e-7, 'cross_periodic_var')
        + Matern52Kernel(h(2.0, 'cross_ls')) * h(2.5e-7, 'cross_var')
    )
    gp = GaussianProcess(SharedIndependentKernel(AutoCrossKernel(k_auto, k_cross, num_outputs=2)))

    frequency = prf.Frequency(start=10, stop=500, npoints=npoints, unit='MHz')
    truth = DatasheetLine(zn=50.0, vf=0.69, k1=2.0, k2=0.05, length=1.0)
    observed = truth.s(frequency)
    model = DatasheetLine(zn=50.5, vf=0.7, k1=2.1, k2=0.05, length=1.0)

    mll = MarginalLogLikelihood(
        predictor=lambda m, f: m.s(f),
        observed=observed,
        likelihood=GaussianLikelihood(1e-8),
        discrepancy=gp,
    )
    return mll, model, frequency


def time_call(fn, *args, repeats: int) -> float:
    """Minimum wall time of ``fn(*args)`` over ``repeats`` calls, after one warm-up."""
    jax.block_until_ready(fn(*args))
    best = float('inf')
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - start)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--npoints', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()

    float_mll, model, frequency = build(args.npoints, random=False)
    random_mll, _, _ = build(args.npoints, random=True)

    # Factorize at the covariance shape the likelihood actually factorizes.
    cov = float_mll.predictive_distribution(model, frequency).covariance()
    cholesky = jax.jit(jnp.linalg.cholesky)

    value = eqx.filter_jit(lambda mll, m: mll(m, frequency))
    model_grad = eqx.filter_jit(eqx.filter_value_and_grad(lambda m, mll: mll(m, frequency)))
    def evaluate_pair(mll_and_model):
        mll, m = mll_and_model
        return mll(m, frequency)
    both_grad = eqx.filter_jit(eqx.filter_value_and_grad(evaluate_pair))

    rows = [
        (f'Cholesky of K + s^2 I, shape {cov.shape}', cholesky, (cov,)),
        ('value, float hyperparameters', value, (float_mll, model)),
        ('value, Random hyperparameters', value, (random_mll, model)),
        ('value + grad wrt model, float hyperparameters', model_grad, (model, float_mll)),
        ('value + grad wrt evaluator and model, Random hyperparameters', both_grad, ((random_mll, model),)),
    ]

    print(f'N = {args.npoints}, minimum over {args.repeats} calls after one warm-up')
    width = max(len(label) for label, _, _ in rows)
    for label, fn, fn_args in rows:
        seconds = time_call(fn, *fn_args, repeats=args.repeats)
        print(f'{label:<{width}}  {seconds * 1e3:9.2f} ms')


if __name__ == '__main__':
    main()
