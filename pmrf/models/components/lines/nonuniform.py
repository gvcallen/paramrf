"""
Non-uniform transmission lines.
"""
from typing import Callable, Any, Dict

import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.frequency import Frequency
from pmrf.rf_functions.connections import cascade_s
from pmrf.parameters import Parameter, as_param
from pmrf.models.model import Model
from pmrf.models.components.lines.uniform import RLGCLine

class ProfiledLine(Model, transparent=True):
    r"""
    A non-uniform transmission line defined by an arbitrary profile.

    This model wraps any RLGC transmission line model and allows its parameters
    to vary as a function of length. For example, the line's characteristic impedance
    can be varied exponentially for impedance matching purposes.

    Any line parameters can be uniform across length or follow a spatial profile
    defined by user-provided function (e.g., splines). Both the uniform parameters
    and the coefficients of the profile functions are registered as `Parameter` objects,
    making the profile parameters compatible with fitting and sampling.

    Supported evaluation methods:
      - 'stepped': A discrete cascaded approximation (default).
      - 'riccati': A continuous ODE solver using the Matrix Riccati differential equations.

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import PhysicalLine, ProfiledLine

        def linear_taper(t, start_val, end_val):
            return start_val + (end_val - start_val) * t

        tapered_line = ProfiledLine(
            PhysicalLine,
            linear_taper, 
            length=0.1,
            zn={'start_val': 50.0, 'end_val': 100.0},
            epr=2.2,
            method='riccati',
            options={'rtol': 1e-6, 'atol': 1e-6},
        )
        
        freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')
        s_taper = tapered_line.s(freq)
    """
    # Config
    line_fn: Callable[[Any], RLGCLine] = eqx.field(static=True)
    floating: bool = False
    profile_fns: Dict[str, Callable] = eqx.field(static=True)
    method: str = eqx.field(static=True)
    options: dict = eqx.field(static=True)

    # Parameters and sub-models
    length: Parameter
    profile_params: Dict[str, Dict[str, Parameter]]
    uniform_params: Dict[str, Parameter]

    def __init__(
        self, 
        line_fn: Callable, 
        profile_fn: Callable | None = None,
        *,
        length: Any = 1,
        floating: bool = False,
        method: str = 'stepped',
        options: dict | None = None,
        name: str | None = None,
        z0: complex = 50.0,
        **line_params
    ):
        super().__init__(name=name, z0=z0)
        
        # Defaults
        options = options or dict()
        if method == 'stepped':
            options.setdefault('N', 50)
        elif method == 'riccati':
            options.setdefault('rtol', 1e-5)
            options.setdefault('atol', 1e-5)
            options.setdefault('max_steps', 1000)
            # Add a small offset to avoid evaluating the jacobian at exactly 0.0
            # if the underlying line model has a singularity at length=0
            options.setdefault('dz_eval', 0.0)

        # Members
        self.line_fn = line_fn
        self.floating = floating
        self.length = as_param(length)
        self.method = method
        self.options = options

        # 1. Instantiate a dummy model to extract all default parameters.
        safe_kwargs = {
            k: v for k, v in line_params.items() 
            if not isinstance(v, (dict, tuple))
        }
        safe_kwargs['length'] = self.length
        
        dummy_model = line_fn(**safe_kwargs)
        base_params = dummy_model.named_params(include_fixed=True)

        # Extract the parameter objects, excluding 'length'
        merged_params = {
            k: v for k, v in base_params.items() 
            if k != 'length' 
        }
        
        # 2. Overlay the user's raw inputs
        merged_params.update(line_params)

        parsed_profile_fns = {}
        parsed_profile_params = {}
        parsed_uniform_params = {}

        # 3. Parse the unified kwargs into Equinox-friendly PyTrees
        for k, v in merged_params.items():
            if isinstance(v, tuple) and len(v) == 2 and callable(v[0]) and isinstance(v[1], dict):
                parsed_profile_fns[k] = v[0]
                parsed_profile_params[k] = {arg: as_param(val) for arg, val in v[1].items()}
            elif isinstance(v, dict):
                if profile_fn is None:
                    raise ValueError(f"Parameter '{k}' was provided as a dict, but no `profile_fn` was specified.")
                parsed_profile_fns[k] = profile_fn
                parsed_profile_params[k] = {arg: as_param(val) for arg, val in v.items()}
            else:
                parsed_uniform_params[k] = as_param(v)
                
        self.profile_fns = parsed_profile_fns
        self.profile_params = parsed_profile_params
        self.uniform_params = parsed_uniform_params

    @eqx.filter_jit
    def section(self, dz: float, **profiled_params) -> RLGCLine:
        # Create a fresh dict to avoid mutating self.uniform_params
        current_params = dict(self.uniform_params)
        current_params['length'] = dz
        current_params['floating'] = self.floating
        current_params.update(profiled_params)
        return self.line_fn(**current_params)

    def _profiled_params(self, t: float) -> dict:
        """Helper to evaluate all profile functions at normalized position t."""
        evaluated_params = {}
        for p_name, func in self.profile_fns.items():
            f_kwargs = self.profile_params.get(p_name, {})
            evaluated_params[p_name] = func(t, **f_kwargs)
        return evaluated_params

    @eqx.filter_jit
    def s(self, freq: Frequency) -> jnp.ndarray:
        if self.method == 'stepped':
            return self._s_stepped(freq)
        elif self.method == 'riccati':
            raise NotImplementedError("Riccati method for ProfileLine not yet functional")
            return self._s_riccati(freq)
        else:
            raise ValueError(f"Unknown evaluation method: '{self.method}'")        
        
    @eqx.filter_jit
    def _s_stepped(self, freq: Frequency) -> jnp.ndarray:
        N = self.options['N']
        dz = self.length / N
        t_centers = (jnp.arange(N) + 0.5) / N
        
        def s_section(t):
            profiled_params = self._profiled_params(t)
            sec = self.section(dz, **profiled_params)
            return sec.s(freq), jnp.asarray(sec.z0, dtype=complex)

        batch_s, batch_z0 = jax.vmap(s_section)(t_centers)
        S_cas, z0_cas = cascade_s(batch_s, batch_z0)
        return S_cas

    @eqx.filter_jit
    def _s_riccati(self, freq: Frequency) -> jnp.ndarray:
        import diffrax

        def s_derivative(t, S_state_tuple, args):
            # 1. Reconstruct complex S-matrix from the real/imag tuple state
            # This explicitly resolves the Diffrax complex dtype warning
            S_real, S_imag = S_state_tuple
            S_state = S_real + 1j * S_imag

            # S_state shape: (..., 2N, 2N)
            N_total = S_state.shape[-1]
            N_ports = N_total // 2

            # Unpack aggregated S-matrix blocks
            S11 = S_state[..., :N_ports, :N_ports]
            S12 = S_state[..., :N_ports, N_ports:]
            S21 = S_state[..., N_ports:, :N_ports]
            S22 = S_state[..., N_ports:, N_ports:]

            profiled_params = self._profiled_params(t)

            # Wrap the line construction to isolate `length`
            def local_s_matrix(dz):
                sec = self.section(dz, **profiled_params)
                return sec.s(freq)

            # 2. Extract Generator Matrix Q(z) = dS/dz using Finite Difference
            # This bypasses the jax.jacfwd crash caused by complex matrix inversions/conjugations 
            # inside the renormalize_s power-wave formulation.
            dz_delta = 1e-8
            S_plus = local_s_matrix(dz_delta)
            S_zero = local_s_matrix(0.0)
            Q = (S_plus - S_zero) / dz_delta

            Q11 = Q[..., :N_ports, :N_ports]
            Q12 = Q[..., :N_ports, N_ports:]
            Q21 = Q[..., N_ports:, :N_ports]
            Q22 = Q[..., N_ports:, N_ports:]

            # Differential Redheffer Star Product
            dS11 = S12 @ Q11 @ S21
            dS12 = S12 @ (Q12 + Q11 @ S22)
            dS21 = (Q21 + S22 @ Q11) @ S21
            dS22 = Q22 + Q21 @ S22 + S22 @ Q12 + S22 @ Q11 @ S22

            # Reconstruct block matrix 
            top = jnp.concatenate([dS11, dS12], axis=-1)
            bot = jnp.concatenate([dS21, dS22], axis=-1)
            dS_dt = jnp.concatenate([top, bot], axis=-2)
            
            dS_dt_scaled = self.length * dS_dt

            # Return as a tuple of real/imag to maintain Diffrax stability
            return jnp.real(dS_dt_scaled), jnp.imag(dS_dt_scaled)

        # Establish Initial Condition S(0) = [0, I; I, 0]
        profiled_params_init = self._profiled_params(0.0)
        sec_dummy = self.section(0.0, **profiled_params_init)
        S_dummy = sec_dummy.s(freq)
        
        batch_shape = S_dummy.shape[:-2]
        N_ports = S_dummy.shape[-1] // 2
        
        I_mat = jnp.broadcast_to(jnp.eye(N_ports, dtype=complex), batch_shape + (N_ports, N_ports))
        Z_mat = jnp.zeros_like(I_mat)
        
        top_init = jnp.concatenate([Z_mat, I_mat], axis=-1)
        bot_init = jnp.concatenate([I_mat, Z_mat], axis=-1)
        S_initial_complex = jnp.concatenate([top_init, bot_init], axis=-2)

        # Package initial state as a real/imaginary tuple
        y0 = (jnp.real(S_initial_complex), jnp.imag(S_initial_complex))

        # Setup and solve ODE
        term = diffrax.ODETerm(s_derivative)
        solver = diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(
            rtol=self.options['rtol'], 
            atol=self.options['atol']
        )

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=0.1, 
            y0=y0,
            stepsize_controller=stepsize_controller,
            max_steps=self.options['max_steps']
        )

        # Unpack the final tuple state back into a standard complex S-matrix array
        S_final_real, S_final_imag = solution.ys[0][-1], solution.ys[1][-1]
        return S_final_real + 1j * S_final_imag