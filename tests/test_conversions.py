"""
Tests for RF parameter conversions ensuring mathematical round-trips,
transitivity, and JAX batching functionality.
"""

import pytest
import jax.numpy as jnp

from pmrf.rf.conversions import (
    s2s, a2s, s2a, s2y, y2s, s2z, z2s,
    y2z, z2y, a2y, y2a, a2z, z2a,
    y2mna, z2mna, a2mna, s2mna, renormalize_s
)

# --- FIXTURES ---

@pytest.fixture
def base_y():
    """Provides a stable, passive 2-port admittance matrix to avoid singularities."""
    return jnp.array([
        [2.0 + 2.0j, -1.0 - 1.0j],
        [-1.0 - 1.0j,  2.0 + 2.0j]
    ])

@pytest.fixture
def z0_scalar():
    return 50.0

@pytest.fixture
def z0_array():
    """Asymmetric characteristic impedances for port 1 and port 2."""
    return jnp.array([50.0, 75.0])


# --- 1. ROUND TRIP TESTS (A -> B -> A) ---

DIRECT_ROUND_TRIP_PAIRS = [
    (y2z, z2y),
    (y2a, a2y),
    (z2a, a2z),
]

@pytest.mark.parametrize("forward_func, backward_func", DIRECT_ROUND_TRIP_PAIRS)
def test_direct_round_trips(base_y, forward_func, backward_func):
    """Tests parameter pairs that don't depend on characteristic impedance."""
    intermediate = forward_func(base_y)
    reverted = backward_func(intermediate)
    assert jnp.allclose(base_y, reverted, atol=1e-6)

S_ROUND_TRIP_PAIRS = [
    (y2s, s2y),
    (z2s, s2z),
]

@pytest.mark.parametrize("to_s, from_s", S_ROUND_TRIP_PAIRS)
@pytest.mark.parametrize("s_def", ["power", "traveling"])
def test_s_param_round_trips(base_y, to_s, from_s, s_def, z0_scalar):
    """Tests conversions into and out of the S-domain across definitions."""
    # Start from Z if testing Z<->S, otherwise start from Y
    base_mat = y2z(base_y) if to_s == z2s else base_y

    s_mat = to_s(base_mat, z0=z0_scalar, s_def=s_def)
    reverted = from_s(s_mat, z0=z0_scalar, s_def=s_def)
    assert jnp.allclose(base_mat, reverted, atol=1e-6)

def test_a2s_s2a_round_trip(base_y, z0_scalar):
    """Tests ABCD to S round-trips (which inherently assume Power waves)."""
    base_a = y2a(base_y)
    s_mat = a2s(base_a, z0=z0_scalar)
    reverted = s2a(s_mat, z0=z0_scalar)
    assert jnp.allclose(base_a, reverted, atol=1e-6)


# --- TRANSITIVITY (A -> B -> C == A -> C) ---

def test_transitivity(base_y, z0_scalar):
    """Proves that converting Y -> Z -> S is mathematically identical to Y -> S."""
    z_mat = y2z(base_y)
    
    s_from_z = z2s(z_mat, z0=z0_scalar, s_def='power')
    s_from_y = y2s(base_y, z0=z0_scalar, s_def='power')
    
    assert jnp.allclose(s_from_z, s_from_y, atol=1e-6)


# --- S-PARAMETER DEFINITION CONVERSIONS ---

def test_s2s_conversion(base_y, z0_array):
    """Tests the vectorized s2s conversion logic against the direct Y2S derivation."""
    # Start with a power-wave S-matrix
    s_power = y2s(base_y, z0=z0_array, s_def='power')

    # Convert it to traveling
    s_traveling = s2s(s_power, z0=z0_array, s_def_old='power', s_def_new='traveling')

    # Check that it matches generating a traveling S-matrix straight from Y
    s_traveling_direct = y2s(base_y, z0=z0_array, s_def='traveling')

    assert jnp.allclose(s_traveling, s_traveling_direct, atol=1e-6)


# --- BATCHING (VMAP) SUPPORT ---

def test_vmap_batching(base_y, z0_scalar):
    """Ensures that the 3D (nfreqs, nports, nports) shape routing works for all functions."""
    # Create a dummy frequency batch of shape (3, 2, 2)
    base_y_3d = jnp.stack([base_y, base_y * 1.1, base_y * 1.2])

    # Direct Y2Z
    z_3d = y2z(base_y_3d)
    assert z_3d.shape == (3, 2, 2)
    assert jnp.allclose(z_3d[0], y2z(base_y), atol=1e-6)

    # Impedance dependent Y2S
    s_3d = y2s(base_y_3d, z0=z0_scalar)
    assert s_3d.shape == (3, 2, 2)
    assert jnp.allclose(s_3d[1], y2s(base_y * 1.1, z0=z0_scalar), atol=1e-6)


# --- MNA STAMPS ---

@pytest.mark.parametrize("mna_func, get_base", [
    (y2mna, lambda y: y),
    (z2mna, y2z),
    (a2mna, y2a),
])
def test_mna_stamp_shapes(base_y, mna_func, get_base):
    """Verifies that MNA block matrices are generated with the correct dimensions."""
    base_mat = get_base(base_y)
    stamp = mna_func(base_mat)

    # For a 2-port network, the core Y block should always be 2x2
    assert stamp.Y.shape == (2, 2)
    
    # B connects Nodes to Aux variables, C connects Aux to Nodes
    assert stamp.B.shape[0] == 2
    assert stamp.C.shape[1] == 2
    
    # Aux variables must form a square matrix
    assert stamp.B.shape[1] == stamp.C.shape[0] == stamp.D.shape[0] == stamp.D.shape[1]


# --- RENORMALIZATION TESTS ---

@pytest.fixture
def base_s(base_y, z0_scalar):
    """Provides a stable base S-parameter matrix evaluated at 50 ohms."""
    return y2s(base_y, z0=z0_scalar)

@pytest.fixture
def z1_scalar():
    return 75.0

def test_renormalize_identity(base_s, z0_scalar):
    """Tests the early-exit branch when impedances and definitions match."""
    s_renorm = renormalize_s(base_s, z_old=z0_scalar, z_new=z0_scalar)
    assert jnp.allclose(s_renorm, base_s, atol=1e-6)

@pytest.mark.parametrize("method", ["mobius", "hub"])
def test_renormalize_round_trip(base_s, z0_scalar, z1_scalar, method):
    """Ensures renormalizing Z0 -> Z1 -> Z0 recovers the original network perfectly."""
    s_75 = renormalize_s(base_s, z_old=z0_scalar, z_new=z1_scalar, method=method)
    s_reverted = renormalize_s(s_75, z_old=z1_scalar, z_new=z0_scalar, method=method)
    
    assert jnp.allclose(base_s, s_reverted, atol=1e-6)

def test_renormalize_methods_equivalent(base_s, z0_scalar, z1_scalar):
    """Proves the Mobius transform yields the exact same result as the Z-parameter hub."""
    s_mobius = renormalize_s(base_s, z_old=z0_scalar, z_new=z1_scalar, method="mobius")
    s_hub = renormalize_s(base_s, z_old=z0_scalar, z_new=z1_scalar, method="hub")
    
    assert jnp.allclose(s_mobius, s_hub, atol=1e-6)

def test_renormalize_definition_routing(base_s, z0_scalar, z1_scalar):
    """Tests that cross-definition renormalization (Power -> Traveling) routes correctly."""
    # Renormalize AND change definition to traveling in one shot
    s_trav_75 = renormalize_s(
        base_s, z_old=z0_scalar, z_new=z1_scalar, 
        s_def_old='power', s_def_new='traveling'
    )
    
    # Do it manually in two discrete steps to verify
    s_pow_75 = renormalize_s(base_s, z_old=z0_scalar, z_new=z1_scalar, s_def_old='power', s_def_new='power')
    s_trav_75_manual = s2s(s_pow_75, z0=z1_scalar, s_def_old='power', s_def_new='traveling')
    
    assert jnp.allclose(s_trav_75, s_trav_75_manual, atol=1e-6)

def test_renormalize_batching(base_s, z0_scalar, z1_scalar):
    """Ensures 3D frequency batching works correctly through the vmap."""
    s_3d = jnp.stack([base_s, base_s * 0.9 + 0.1j])
    s_renorm_3d = renormalize_s(s_3d, z_old=z0_scalar, z_new=z1_scalar)
    
    assert s_renorm_3d.shape == (2, 2, 2)
    assert jnp.allclose(s_renorm_3d[0], renormalize_s(base_s, z0_scalar, z1_scalar), atol=1e-6)