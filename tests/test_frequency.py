# tests/test_core/test_frequency.py
import signal
from contextlib import contextmanager

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.frequency import Frequency


@contextmanager
def deadline(seconds):
    """Fails the test instead of hanging if the block runs longer than `seconds`."""
    def expire(signum, frame):
        raise TimeoutError(f"did not return within {seconds} s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

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

@pytest.mark.parametrize("index", [0, 9, -1, -10, np.int64(3), np.int32(-2)])
def test_getitem_in_range_int(basic_freq, index):
    """In-range integers, including negative and NumPy ones, select one point."""
    single = basic_freq[index]
    assert single.npoints == 1
    assert single.f[0] == basic_freq.f[int(index)]


@pytest.mark.parametrize("index", [10, -11, 10**6, np.int64(10), np.int32(-11)])
def test_getitem_out_of_range_int_raises(basic_freq, index):
    """Integers outside [-npoints, npoints) raise like any Python sequence."""
    with pytest.raises(IndexError):
        basic_freq[index]


def test_getitem_int_on_empty_raises():
    """An empty axis has no valid integer index."""
    with pytest.raises(IndexError):
        Frequency()[0]


def test_getitem_slice_past_end_clips(basic_freq):
    """Slices clip at the axis ends rather than raising."""
    assert basic_freq[5:100].npoints == 5
    assert basic_freq[20:].npoints == 0


def test_getitem_string(basic_freq):
    """String keys select a band or the nearest point."""
    band = basic_freq['2-4ghz']
    assert jnp.allclose(band.f, jnp.array([2e9, 3e9, 4e9]))
    nearest = basic_freq['3.2']
    assert nearest.npoints == 1
    assert nearest.f[0] == 3e9


def test_getitem_traced_int_under_jit(basic_freq):
    """A traced index is not bounds-checked and still selects a point."""
    @jax.jit
    def pick(freq, i):
        return freq[i].f

    assert pick(basic_freq, 3)[0] == basic_freq.f[3]

# ---------------------------------------------------------
# NumPy Coercion
# ---------------------------------------------------------

def test_asarray_returns_hz():
    """NumPy coercion returns the frequency vector in Hz."""
    freq = Frequency(50, 200, 151, 'MHz')
    with deadline(10):
        arr = np.asarray(freq)
    assert arr.shape == (151,)
    assert arr.dtype.kind == 'f'
    np.testing.assert_array_equal(arr, np.asarray(freq.f))


def test_asarray_dtype(basic_freq):
    """A requested dtype is honoured."""
    with deadline(10):
        arr = np.asarray(basic_freq, dtype=np.float32)
    assert arr.dtype == np.float32


def test_asarray_traced_raises(basic_freq):
    """Coercing a traced `Frequency` raises JAX's tracer-conversion error."""
    @jax.jit
    def coerce(freq):
        return np.asarray(freq)

    with deadline(10), pytest.raises(jax.errors.TracerArrayConversionError):
        coerce(basic_freq)

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


def test_pytree_has_single_f_leaf(basic_freq):
    """The only array leaf is `f`; `unit` stays static."""
    leaves, treedef = jax.tree_util.tree_flatten(basic_freq)
    assert len(leaves) == 1 and leaves[0] is basic_freq.f
    assert jax.tree_util.tree_unflatten(treedef, leaves).unit == 'GHz'


def test_vmap_over_frequency_axis(basic_freq):
    """`vmap` maps over the points of `f`."""
    out = jax.vmap(lambda freq: freq.w)(basic_freq)
    assert jnp.allclose(out, basic_freq.w)

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


def test_skrf_from_f_accepts_frequency():
    """scikit-rf builds its own axis from a `Frequency` without hanging."""
    skrf = pytest.importorskip("skrf")
    freq = Frequency(50, 200, 151, 'MHz')
    with deadline(10):
        skrf_freq = skrf.Frequency.from_f(freq, unit='Hz')
    np.testing.assert_allclose(skrf_freq.f, np.asarray(freq.f))
