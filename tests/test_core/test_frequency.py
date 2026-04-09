# tests/test_core/test_frequency.py
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.core.frequency import Frequency

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def basic_freq():
    """Returns a 1 to 10 GHz frequency band with 10 points."""
    return Frequency(start=1.0, stop=10.0, npoints=10, unit='GHz')

# ---------------------------------------------------------
# Initialization & Constructors
# ---------------------------------------------------------

def test_initialization(basic_freq):
    """Test standard initialization and unit scaling."""
    assert basic_freq.unit == 'GHz'
    assert basic_freq.npoints == 10
    assert basic_freq.start == 1e9
    assert basic_freq.stop == 10e9
    assert basic_freq.f.shape == (10,)

def test_from_f_scalar():
    """Test initialization from a scalar frequency."""
    freq = Frequency.from_f(5.0, unit='MHz')
    assert freq.npoints == 1
    assert freq.start == 5e6
    assert freq.f.shape == (1,)

def test_from_f_array():
    """Test initialization from a JAX array."""
    f_arr = jnp.array([1.0, 2.0, 3.0])
    freq = Frequency.from_f(f_arr, unit='kHz')
    assert freq.npoints == 3
    assert freq.start == 1e3
    assert freq.stop == 3e3
    assert freq.unit == 'kHz'

# ---------------------------------------------------------
# Properties & Math
# ---------------------------------------------------------

def test_scaled_properties(basic_freq):
    """Test that scaled properties return values in the specified unit."""
    assert basic_freq.start_scaled == pytest.approx(1.0)
    assert basic_freq.stop_scaled == pytest.approx(10.0)
    assert basic_freq.span_scaled == pytest.approx(9.0)
    assert basic_freq.center_scaled == pytest.approx(5.5)
    assert jnp.allclose(basic_freq.f_scaled, jnp.linspace(1.0, 10.0, 10))

def test_w_property(basic_freq):
    """Test angular frequency conversion."""
    expected_w = 2 * jnp.pi * basic_freq.f
    assert jnp.allclose(basic_freq.w, expected_w)

def test_gradients(basic_freq):
    """Test that frequency gradients (df, dw) compute without error."""
    assert basic_freq.df.shape == basic_freq.f.shape
    assert basic_freq.dw.shape == basic_freq.w.shape

# ---------------------------------------------------------
# Slicing and Indexing
# ---------------------------------------------------------

def test_getitem_int_and_slice(basic_freq):
    """Test standard python/JAX indexing on the Frequency object."""
    # Integer index
    single_freq = basic_freq[0]
    assert single_freq.npoints == 1
    assert single_freq.start == 1e9

    # Slice index
    sliced_freq = basic_freq[0:5]
    assert sliced_freq.npoints == 5
    assert jnp.allclose(sliced_freq.f, basic_freq.f[0:5])

# ---------------------------------------------------------
# Arithmetic Operators
# ---------------------------------------------------------

def test_math_operators():
    """Test elementwise operator overloading for Frequency objects."""
    f1 = Frequency.from_f([1.0, 2.0], unit='Hz')
    f2 = Frequency.from_f([3.0, 4.0], unit='Hz')

    # Addition
    f_add = f1 + f2
    assert jnp.allclose(f_add.f, jnp.array([4.0, 6.0]))

    # Subtraction
    f_sub = f2 - f1
    assert jnp.allclose(f_sub.f, jnp.array([2.0, 2.0]))

    # Multiplication (with scalar broadcast)
    f_mul = f1 * 10.0
    assert jnp.allclose(f_mul.f, jnp.array([10.0, 20.0]))

    # Division
    f_div = f2 / f1
    assert jnp.allclose(f_div.f, jnp.array([3.0, 2.0]))

# ---------------------------------------------------------
# PyTree & JAX Compatibility
# ---------------------------------------------------------

def test_jax_jit_compatibility(basic_freq):
    """
    Ensures that the Frequency object can pass through JAX boundaries.
    Because `unit` is field(static=True), it should compile perfectly
    without raising 'non-array argument' TypeErrors.
    """
    @jax.jit
    def compute_w(freq_obj):
        # We perform an operation on the internal JAX array
        return freq_obj.w

    # If PyTree flattening fails or static fields aren't respected, this throws an error.
    w_out = compute_w(basic_freq)
    assert w_out.shape == (10,)

# ---------------------------------------------------------
# scikit-rf Interoperability (Conditional)
# ---------------------------------------------------------

def test_skrf_interop():
    """Test conversion to and from scikit-rf Frequency objects, if installed."""
    skrf = pytest.importorskip("skrf")
    
    # Create scikit-rf frequency
    skrf_freq = skrf.Frequency(1, 10, 10, 'ghz')
    
    # Convert to ParamRF
    pmrf_freq = Frequency.from_skrf(skrf_freq)
    assert pmrf_freq.unit == 'GHz'
    assert pmrf_freq.npoints == 10
    assert pmrf_freq.start == 1e9

    # Convert back to scikit-rf
    back_to_skrf = pmrf_freq.to_skrf()
    assert isinstance(back_to_skrf, skrf.Frequency)
    assert back_to_skrf.unit.lower() == 'ghz'