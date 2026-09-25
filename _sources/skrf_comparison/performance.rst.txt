Performance
=====================

Because **ParamRF** is built on top of JAX, models are not evaluated using standard Python loops. Instead, circuits are compiled into highly optimized XLA (Accelerated Linear Algebra) computational graphs, allowing natively batched matrix operations that can be dispatched directly to your CPU or GPU. Since the same graph is differentiable, this advantage extends to gradient-based optimization as well, where ``scikit-rf`` has to fall back on finite differences.

To illustrate both cases on a single example, this section benchmarks a 25-section microstrip transformer which matches a 50 :math:`\Omega` source to a 200 :math:`\Omega` load. Although this is a somewhat contrived example due to the load being fixed (usually an analytical solution is easily available for such matching case), it demonstrates the performance gain that is possible for more complex, frequency-dependent case where line sections containing on the order of 50 free parameters are possible (in this case, each line's characteristic impedance and physical length are treated as free).

Both a single forward-pass evaluation of the circuit and the end-to-end optimization timings are compared.

Benchmark Script
-----------------
The script below builds the same circuit in ``scikit-rf`` and ParamRF once, then reuses the two builder functions for both benchmarks. In an optimization context, physical parameters change at every step, so a fair comparison must reconstruct the circuit topology and recompute the physical media properties on every evaluation, which both benchmarks do.

.. code-block:: python

    import time
    import numpy as np
    import scipy.constants
    import skrf as rf
    import jax
    import equinox as eqx
    import parax as prx
    import pmrf as prf
    from skrf.circuit import Circuit as CircuitSkrf
    from scipy.optimize import minimize as scipy_minimize, Bounds
    from pmrf.models import (
        Circuit as CircuitPmrf, Port, Ground, Resistor, PhysicalLine,
        GlobalMNACircuitSolver,
    )
    from pmrf.parameters import Bounded
    from pmrf.evaluators import Goal
    from pmrf.optimize import minimize as pmrf_minimize, ScipyMinimize

    c = scipy.constants.c

    # An N-section microstrip transformer matching a 50 ohm source to a
    # 200 ohm load. Each section contributes two free parameters: its
    # characteristic impedance and its physical length.
    N_SECTIONS = 25
    EPR, ATTEN_A, ATTEN_FA, TAND = 4.4, 0.05, 1e9, 0.02   # substrate parameters
    Z0, ZL = 50.0, 200.0
    F_START, F_STOP, N_POINTS = 2.0, 4.0, 201             # GHz
    TARGET_DB = -18.0

    freq_skrf = rf.Frequency(start=F_START, stop=F_STOP, npoints=N_POINTS, unit='ghz')
    freq_pmrf = prf.Frequency(start=F_START, stop=F_STOP, npoints=N_POINTS, unit='ghz')

    def create_skrf_physical_line(freq, zn, length, ep_r, A, f_A, tand, name):
        """Builds a scikit-rf line matching ParamRF's PhysicalLine physics exactly."""
        f = freq.f
        sqrt_ep_r = np.sqrt(ep_r)
        A_dB = A * np.sqrt(f / f_A)
        alpha_c = A_dB * (np.log(10) / 20.0)
        alpha_d = np.pi * sqrt_ep_r * f / c * tand
        R = 2 * zn * alpha_c
        L = (zn * sqrt_ep_r) / c
        G = 2 / zn * alpha_d
        C = sqrt_ep_r / (zn * c)
        omega = 2 * np.pi * f
        Z_series = R + 1j * omega * L
        Y_shunt = G + 1j * omega * C
        gamma = np.sqrt(Z_series * Y_shunt)
        Zc = np.sqrt(Z_series / Y_shunt)
        media = rf.media.DefinedGammaZ0(frequency=freq, gamma=gamma, z0=Zc)
        return media.line(d=length, unit='m', name=name)

    def build_skrf_transformer(zns, lens):
        """
        Rebuilds the topology from scratch."""
        lines = [
            create_skrf_physical_line(freq_skrf, zns[i], lens[i], EPR, ATTEN_A, ATTEN_FA, TAND, f'line{i}')
            for i in range(N_SECTIONS)
        ]
        load = rf.media.DefinedGammaZ0(frequency=freq_skrf, z0=50).resistor(ZL, name='load')
        port0 = CircuitSkrf.Port(freq_skrf, 'p0')
        gnd = CircuitSkrf.Ground(freq_skrf, 'gnd')

        conns = [[(port0, 0), (lines[0], 0)]]
        for i in range(N_SECTIONS - 1):
            conns.append([(lines[i], 1), (lines[i + 1], 0)])
        conns.append([(lines[N_SECTIONS - 1], 1), (load, 0)])
        conns.append([(load, 1), (gnd, 0)])
        return CircuitSkrf(conns).network

    def build_pmrf_transformer(zns, lens):
        """zns/lens hold either plain floats (fixed) or pmrf.Bounded instances (free parameters)."""
        p0, gnd = Port(), Ground()
        lines = [
            PhysicalLine(zn=zns[i], length=lens[i], ep_r=EPR, A=ATTEN_A, f_A=ATTEN_FA, tand=TAND)
            for i in range(N_SECTIONS)
        ]
        load = Resistor(R=ZL)
        conns = [[(p0, 0), (lines[0], 0)]]
        for i in range(N_SECTIONS - 1):
            conns.append([(lines[i], 1), (lines[i + 1], 0)])
        conns.append([(lines[N_SECTIONS - 1], 1), (load, 0)])
        conns.append([(load, 1), (gnd, 0)])
        return CircuitPmrf(connections=conns, solver=GlobalMNACircuitSolver(), flatten=True)

    # Starting point: every section set to the impedance of a single-section
    # quarter-wave transformer, before any per-section tuning.
    ZN0 = np.full(N_SECTIONS, np.sqrt(Z0 * ZL))
    LEN0 = np.full(N_SECTIONS, c / (4 * 3e9 * np.sqrt(EPR)))

    def run_forward_pass_benchmark():
        num_runs = 200

        _ = build_skrf_transformer(ZN0, LEN0)
        t0 = time.perf_counter()
        for _ in range(num_runs):
            build_skrf_transformer(ZN0, LEN0)
        t_skrf_ms = (time.perf_counter() - t0) / num_runs * 1000

        circuit_model = build_pmrf_transformer(ZN0, LEN0)
        is_dynamic = lambda x: eqx.is_inexact_array(x) and not isinstance(x, np.ndarray)
        params, static_model = eqx.partition(circuit_model, is_dynamic, is_leaf=prx.is_constant)

        def eval_pmrf(p):
            model = eqx.combine(p, static_model, is_leaf=prx.is_constant)
            return prx.unwrap(model).s(freq_pmrf)

        jitted_pmrf = jax.jit(eval_pmrf)
        _ = jitted_pmrf(params).block_until_ready()  # AOT compilation / warmup

        t0 = time.perf_counter()
        for _ in range(num_runs):
            jitted_pmrf(params).block_until_ready()
        t_pmrf_ms = (time.perf_counter() - t0) / num_runs * 1000

        print(f"scikit-rf forward pass: {t_skrf_ms:6.3f} ms")
        print(f"ParamRF forward pass:   {t_pmrf_ms:6.3f} ms")
        print(f"Speedup:                {t_skrf_ms / t_pmrf_ms:5.1f}x")

    def run_optimization_benchmark():
        zn_bounds, len_bounds = (15.0, 250.0), (0.002, 0.030)
        max_iter = 300

        def hinge_mse(s11_db):
            return np.mean(np.maximum(s11_db - TARGET_DB, 0.0) ** 2)

        def s11_db_skrf(x):
            s11 = build_skrf_transformer(x[:N_SECTIONS], x[N_SECTIONS:]).s[:, 0, 0]
            return 20 * np.log10(np.abs(s11))

        x0 = np.concatenate([ZN0, LEN0])
        lower = np.concatenate([np.full(N_SECTIONS, zn_bounds[0]), np.full(N_SECTIONS, len_bounds[0])])
        upper = np.concatenate([np.full(N_SECTIONS, zn_bounds[1]), np.full(N_SECTIONS, len_bounds[1])])

        t0 = time.perf_counter()
        skrf_result = scipy_minimize(
            lambda x: hinge_mse(s11_db_skrf(x)), x0, jac='2-point', method='trust-constr',
            bounds=Bounds(lower, upper), options={'maxiter': max_iter},
        )
        t_skrf = time.perf_counter() - t0

        pmrf_model = build_pmrf_transformer(
            [Bounded(*zn_bounds, value=v) for v in ZN0],
            [Bounded(*len_bounds, value=v) for v in LEN0],
        )
        goal = Goal('s11_db', '<', TARGET_DB, loss='mse')

        t0 = time.perf_counter()
        pmrf_result = pmrf_minimize(
            goal, pmrf_model, freq_pmrf,
            solver=ScipyMinimize(method='trust-constr', show_progress=False),
            max_iter=max_iter,
        )
        t_pmrf = time.perf_counter() - t0

        print(f"scikit-rf + SciPy (finite differences): {t_skrf:6.1f} s  ({skrf_result.nit} iterations, {skrf_result.nfev} evaluations)")
        print(f"ParamRF + SciPy (analytic gradients):   {t_pmrf:6.1f} s  ({pmrf_result.metrics.nit} iterations, {pmrf_result.metrics.nfev} evaluations)")
        print(f"Speedup:                                {t_skrf / t_pmrf:5.1f}x")

        max_s11_skrf = np.max(s11_db_skrf(skrf_result.x))
        max_s11_pmrf = np.max(np.array(pmrf_result.model.s_db(freq_pmrf))[:, 0, 0])
        print(f"\nWorst-case |S11| after {max_iter} iterations (target < {TARGET_DB} dB):")
        print(f"  scikit-rf: {max_s11_skrf:5.2f} dB")
        print(f"  ParamRF:   {max_s11_pmrf:5.2f} dB")

    if __name__ == "__main__":
        run_forward_pass_benchmark()
        print()
        run_optimization_benchmark()

Expected Output
----------------
Running this script on a standard modern CPU yields results similar to the following. The optimization benchmark takes on the order of ten minutes, since ``scikit-rf`` has to re-simulate the circuit close to 20000 times to complete 300 finite-difference iterations.

.. code-block:: text

    scikit-rf forward pass: 39.359 ms
    ParamRF forward pass:    1.197 ms
    Speedup:                 32.9x

    scikit-rf + SciPy (finite differences):  779.9 s  (300 iterations, 19278 evaluations)
    ParamRF + SciPy (analytic gradients):      4.2 s  (300 iterations, 335 evaluations)
    Speedup:                                 187.3x

    Worst-case |S11| after 300 iterations (target < -18.0 dB):
      scikit-rf: -17.57 dB
      ParamRF:   -18.00 dB

For a single evaluation, ParamRF is roughly 33x faster: the whole transformer is represented as one compiled XLA graph, while ``scikit-rf`` rebuilds and connects 25 separate Python network objects on every call.

The speedup experienced is even larger for the end-to-end optimization. Both optimizers start from the same point and run for the same 300-iteration budget. However, computing the gradient of a 50-parameter model with finite differences requires 51 circuit evaluations per step (one baseline plus one perturbation per parameter), so ``scikit-rf``'s cost per iteration grows with the number of parameters. ParamRF instead traces the whole circuit once and obtains the exact gradient for all 50 parameters from a single, compiled forward-and-backward pass via :func:`jax.value_and_grad`, rather than 51 separate forward passes. In the same number of iterations, ParamRF reaches the -18 dB design target, while ``scikit-rf`` is still short of it, since its noisier finite-difference gradients make less progress per step.
