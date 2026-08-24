# tests/test_adapters/test_adapters.py
import pytest
import warnings
import jax
import jax.numpy as jnp
import numpy as np
import parax as prx

import pmrf as prf
from pmrf import Frequency, Param, param
from pmrf.models import (
    AbstractBuilder, AbstractDiscrete, AbstractSingleDomain,
    AbstractHost, Cascade, ContinuousCallable, PhaseLine, SkrfNetwork,
    Touchstone,
)
from pmrf.network_collection import NetworkCollection
from pmrf.types import ArrayLike

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def coarse_freq():
    # 3 points: 1, 2, 3 GHz
    return Frequency(start=1.0, stop=3.0, npoints=3, unit='GHz')

@pytest.fixture
def fine_freq():
    # 5 points: 1, 1.5, 2.0, 2.5, 3.0 GHz
    return Frequency(start=1.0, stop=3.0, npoints=5, unit='GHz')


class BuiltLine(AbstractBuilder):
    theta: Param = param()

    def build(self):
        return PhaseLine(z0=55.0, theta=self.theta, f0=5e9)


class BuiltCascade(AbstractBuilder):
    theta: Param = param()

    def build(self):
        return Cascade((
            PhaseLine(z0=50.0, theta=self.theta, f0=5e9),
            PhaseLine(z0=50.0, theta=10.0, f0=5e9),
        ))


class InvalidBuilder(AbstractBuilder):
    def build(self):
        return "not a model"


def test_abstract_builder_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractBuilder()


def test_abstract_builder_delegates_complete_rf_interface(fine_freq):
    builder = BuiltLine(theta=25.0)
    built = builder.build()

    assert builder.number_of_ports == built.number_of_ports == 2
    assert builder.nports == built.nports
    assert builder.primary_domain == built.primary_domain
    assert jnp.allclose(builder.primary_matrix(fine_freq), built.primary_matrix(fine_freq))
    assert jnp.allclose(builder.s(fine_freq, z0=63.0), built.s(fine_freq, z0=63.0))
    assert jnp.allclose(builder.a(fine_freq), built.a(fine_freq))
    assert jnp.allclose(builder.y(fine_freq), built.y(fine_freq))
    assert jnp.allclose(builder.z(fine_freq), built.z(fine_freq))

    builder_stamp = builder.mna(fine_freq)
    built_stamp = built.mna(fine_freq)
    assert all(
        jnp.allclose(actual, expected)
        for actual, expected in zip(
            jax.tree.leaves(builder_stamp), jax.tree.leaves(built_stamp)
        )
    )


def test_abstract_builder_delegates_expand():
    builder = BuiltCascade(theta=25.0)
    expanded = builder.expand()

    assert expanded is not None
    port_map, internal_connections = expanded
    assert len(port_map) == builder.nports == 2
    assert internal_connections == []


def test_abstract_builder_build_does_not_warn():
    builder = BuiltLine(theta=25.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert isinstance(builder.build(), prf.Model)

    assert not any(issubclass(w.category, FutureWarning) for w in caught)


def test_abstract_builder_validates_build_result():
    with pytest.raises(TypeError, match=r"build\(\) must return a pmrf\.Model"):
        InvalidBuilder().s(Frequency(1.0, 2.0, 2, unit="GHz"))


def test_abstract_builder_tracks_functional_parameter_updates(fine_freq):
    builder = BuiltLine(theta=10.0)
    updated = builder.at("theta").set(prf.as_param(35.0))

    assert not jnp.allclose(builder.s(fine_freq), updated.s(fine_freq))

    derivative = jax.grad(
        lambda theta: jnp.real(BuiltLine(theta=theta).s(fine_freq)[0, 1, 0])
    )(25.0)
    assert jnp.isfinite(derivative)
    assert not jnp.isclose(derivative, 0.0)

# ---------------------------------------------------------
# Abstract Adapter Dummies & Tests
# ---------------------------------------------------------

class DummyDiscrete(AbstractDiscrete):
    """A 1-port discrete model with tabulated S-parameters."""
    frequency: Frequency

    def s_discrete(self, z0: ArrayLike = 50.0) -> jnp.ndarray:
        vals = jnp.array([1.0, 2.0, 3.0], dtype=complex)
        return vals.reshape(-1, 1, 1)

def test_abstract_discrete_interpolation(coarse_freq, fine_freq):
    """Ensure the discrete model interpolates correctly when queried at new frequencies."""
    model = DummyDiscrete(frequency=coarse_freq)
    
    # Query at the new, finer frequency grid
    s_interp = model.s(fine_freq)
    
    assert s_interp.shape == (5, 1, 1)
    # The middle point (1.5 GHz) should interpolate perfectly to 1.5
    assert jnp.allclose(s_interp[1, 0, 0], 1.5 + 0.0j)
    
class DummySinglePropertyY(AbstractSingleDomain):
    """A model that only natively knows its Y-parameters."""
    domain: str = 'y'
    def matrix(self, freq: Frequency) -> jnp.ndarray:
        return jnp.ones((freq.npoints, 1, 1), dtype=complex) * 0.02 # 50 ohm admittance

def test_single_property_routing(fine_freq):
    """Test that specifying property='y' correctly routes and triggers conversions."""
    model = DummySinglePropertyY()
    
    # Querying Y should hit matrix() directly
    y_mat = model.y(fine_freq)
    assert jnp.allclose(y_mat, 0.02)
    
    # Querying S should trigger y2s conversion natively
    s_mat = model.s(fine_freq)
    # 50 ohm admittance in a 50 ohm system -> matched (S11 = 0)
    assert jnp.allclose(s_mat, 0.0 + 0.0j)

# ---------------------------------------------------------
# Host Model Tests
# ---------------------------------------------------------

class DummyHostModel(AbstractHost):
    """A dummy host model representing an external simulator."""
    val: Param = param(default=10.0, as_free=True)
    
    @property
    def number_of_ports(self): 
        return 1
    
    def compute(self, freq: Frequency) -> np.ndarray:
        # Safely extract scalar from val (handles Parameters, JAX arrays, and Numpy arrays)
        v = float(np.array(getattr(self.val, 'value', self.val)).item())
        nf = freq.npoints
        return np.ones((nf, 1, 1), dtype=complex) * v

def test_host_model_single_execution(fine_freq):
    """Test standard single-thread execution of a Host model."""
    model = DummyHostModel(val=5.0, domain='s')
    s = model.s(fine_freq)
    
    assert s.shape == (5, 1, 1)
    assert jnp.allclose(s, 5.0 + 0.0j)

def test_host_model_vmap_multithreading(fine_freq):
    """Test that Host models successfully map batched parameters using the ThreadPool."""
    # Create a batch of 3 parameter values as a JAX array
    batched_val = jnp.array([1.0, 2.0, 3.0])
    model = DummyHostModel(val=batched_val)
    
    # Build an in_axes tree matching the model structure.
    # Map over JAX arrays (dynamic/batched), but skip NumPy arrays (static) and scalars.
    axes = jax.tree.map(
        lambda x: 0 if isinstance(x, jax.Array) and x.ndim > 0 else None, 
        model
    )
    
    # Pass the custom in_axes to vmap.
    run_batch = jax.vmap(lambda m: m.s(fine_freq), in_axes=(axes,))
    
    # Run the batched execution
    s_batch = run_batch(model)
    
    # Output should be (batch=3, nfreq=5, nports=1, nports=1)
    assert s_batch.shape == (3, 5, 1, 1)
    assert jnp.allclose(s_batch[0, 0, 0, 0], 1.0 + 0.0j)
    assert jnp.allclose(s_batch[2, 0, 0, 0], 3.0 + 0.0j)

# ---------------------------------------------------------
# Callable Adapter Tests
# ---------------------------------------------------------

def test_continuous_callable(fine_freq):
    """Test wrapping a standard mathematical python function."""
    def dummy_fn(f_scaled, theta):
        # theta is a (1,) array, f_scaled is (5,)
        # Create a dummy S11 matching the frequency
        return (theta[0] * f_scaled).reshape(-1, 1, 1)
        
    model = ContinuousCallable(
        fn=dummy_fn,
        theta=[jnp.array(2.0)], 
    )
    
    s = model.s(fine_freq)
    # At 1.5 GHz, 2.0 * 1.5 = 3.0
    assert jnp.allclose(s[1, 0, 0], 3.0 + 0.0j)

# ---------------------------------------------------------
# Measured & NetworkCollection Tests
# ---------------------------------------------------------

def test_measured_skrf_interpolation(coarse_freq, fine_freq):
    """Test wrapping a scikit-rf Network and interpolating its data."""
    skrf = pytest.importorskip("skrf")
    
    skrf_freq = coarse_freq.to_skrf()
    s_data = np.array([1.0, 2.0, 3.0]).reshape(-1, 1, 1)
    ntwk = skrf.Network(frequency=skrf_freq, s=s_data, z0=50)
    
    measured_model = SkrfNetwork(network=ntwk)
    s_interp = measured_model.s(fine_freq)
    
    assert s_interp.shape == (5, 1, 1)
    assert jnp.allclose(s_interp[1, 0, 0], 1.5 + 0.0j)


def _complex_multiport_network():
    skrf = pytest.importorskip("skrf")
    frequency = skrf.Frequency(1.0, 7.0, 7, unit="GHz")
    x = np.linspace(-1.0, 1.0, frequency.npoints)
    s = np.empty((frequency.npoints, 2, 2), dtype=complex)
    s[:, 0, 0] = 0.12 + 0.04*x - 0.03*x**2 + 0.02j*(x**3 - x)
    s[:, 0, 1] = 0.55 - 0.08*x**3 + 0.03j*(x + x**2)
    s[:, 1, 0] = -0.18 + 0.05*x**2 + 0.04j*(x**3 + x)
    s[:, 1, 1] = 0.08*x**3 - 0.02j*(1.0 - x**2)
    return skrf.Network(frequency=frequency, s=s, z0=50)


def test_skrf_cubic_complex_multiport_matches_skrf_and_source_knots():
    network = _complex_multiport_network()
    model = SkrfNetwork(network, interpolation_kind="cubic")

    knot_frequency = Frequency.from_skrf(network.frequency)
    np.testing.assert_allclose(
        np.asarray(model.s(knot_frequency)),
        model.network.s,
        rtol=2e-6,
        atol=2e-6,
    )

    requested = Frequency.from_f(
        [1.0, 1.35, 2.4, 3.7, 5.25, 6.6, 7.0], unit="GHz"
    )
    expected = model.network.interpolate(
        requested.to_skrf(), kind="cubic"
    ).s
    np.testing.assert_allclose(
        np.asarray(model.s(requested)), expected, rtol=2e-5, atol=2e-6
    )


def test_skrf_cubic_is_jittable_and_nan_outside_source_range():
    model = SkrfNetwork(
        _complex_multiport_network(), interpolation_kind="cubic"
    )
    requested = Frequency.from_f([0.5, 1.5, 4.25, 7.5], unit="GHz")

    result = jax.jit(lambda frequency: model.s(frequency))(requested)

    assert result.shape == (4, 2, 2)
    assert jnp.all(jnp.isnan(result[jnp.array([0, -1])]))
    assert jnp.all(jnp.isfinite(result[1:3]))


def test_skrf_cubic_coefficients_are_fixed_numpy_data():
    model = SkrfNetwork(
        _complex_multiport_network(), interpolation_kind="cubic"
    )

    assert isinstance(model._spline_coefficients, prx.Static)
    shape, dtype, buffer = model._spline_coefficients.unwrap()
    coefficients = np.frombuffer(buffer, dtype=np.dtype(dtype)).reshape(shape)
    assert isinstance(coefficients, np.ndarray)
    assert np.iscomplexobj(coefficients)
    assert model.named_params() == {}


def test_skrf_cubic_preserves_impedance_renormalization():
    network = _complex_multiport_network()
    network.renormalize(63.0, "power")
    model = SkrfNetwork(network, interpolation_kind="cubic")
    requested = Frequency.from_f([1.4, 2.75, 4.5, 6.25], unit="GHz")

    expected = model.network.interpolate(requested.to_skrf(), kind="cubic")
    expected.renormalize(75.0, "power")

    np.testing.assert_allclose(
        np.asarray(model.s(requested, z0=75.0)),
        expected.s,
        rtol=3e-5,
        atol=3e-6,
    )


def test_skrf_interpolation_kind_validation_and_linear_default():
    network = _complex_multiport_network()
    requested = Frequency.from_f([1.5, 2.5, 4.5, 6.5], unit="GHz")

    default_result = SkrfNetwork(network).s(requested)
    explicit_result = SkrfNetwork(
        network, interpolation_kind="linear"
    ).s(requested)
    expected = network.interpolate(requested.to_skrf(), kind="linear").s

    np.testing.assert_allclose(np.asarray(default_result), expected, atol=2e-6)
    np.testing.assert_allclose(default_result, explicit_result)
    with pytest.raises(ValueError, match="interpolation_kind"):
        SkrfNetwork(network, interpolation_kind="quadratic")


def test_skrf_getattr_preserves_cubic_interpolation_kind():
    network = _complex_multiport_network()
    network.name = "named_network"

    named_model = SkrfNetwork(network, interpolation_kind="cubic")
    assert named_model.named_network.interpolation_kind == "cubic"

    collection_model = SkrfNetwork(
        NetworkCollection([network]), interpolation_kind="cubic"
    )
    assert collection_model.named_network.interpolation_kind == "cubic"


def test_touchstone_forwards_interpolation_kind(tmp_path):
    network = _complex_multiport_network()
    path = tmp_path / "complex_multiport.s2p"
    network.write_touchstone(path)

    model = Touchstone(str(path), interpolation_kind="cubic")

    assert model.interpolation_kind == model.touchstone.interpolation_kind == "cubic"
    requested = Frequency.from_f([1.5, 3.25, 6.5], unit="GHz")
    expected = model.touchstone.network.interpolate(
        requested.to_skrf(), kind="cubic"
    ).s
    np.testing.assert_allclose(
        np.asarray(model.s(requested)), expected, rtol=2e-5, atol=2e-6
    )

def test_measured_network_collection_getattr(coarse_freq):
    """Test dynamic attribute access for NetworkCollections."""
    skrf = pytest.importorskip("skrf")
    skrf_f = coarse_freq.to_skrf()
    
    ntwk1 = skrf.Network(frequency=skrf_f, s=np.ones((3,1,1)), z0=50)
    ntwk1.name = 'thru'
    
    ntwk2 = skrf.Network(frequency=skrf_f, s=np.ones((3,1,1))*2, z0=50)
    ntwk2.name = 'line'
    
    nc = NetworkCollection([ntwk1, ntwk2])
    measured_collection = SkrfNetwork(network=nc)
    
    sub_model = measured_collection.thru
    
    assert isinstance(sub_model, SkrfNetwork)
    assert np.allclose(sub_model.network.s, ntwk1.s)
    
    with pytest.raises(Exception, match="Cannot call s\\(\\) on a Measured model that contains a NetworkCollection"):
        measured_collection.s(coarse_freq)
