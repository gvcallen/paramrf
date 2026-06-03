"""tests/test_simulate/test_nodal.py"""

import pytest
import jax.numpy as jnp
import numpy as np

# Adjust imports based on your project structure
from pmrf.simulate.base import NodalRepresentation, MNARepresentation
from pmrf.simulate.solvers.nodal import GlobalNodalReducer, GlobalMNAReducer


def test_global_nodal_reducer_series_admittances():
    """
    Test standard Nodal Admittance Matrix (NAM) reduction.
    
    Circuit: 
    Node 0 (Ext) ---[ Y1 = 2S ]--- Node 1 (Int) ---[ Y2 = 2S ]--- Node 2 (Ext)
    
    Two 2.0 Siemens admittances in series should reduce to a single 1.0 Siemens 
    equivalent admittance between Node 0 and Node 2.
    """
    solver = GlobalNodalReducer()
    
    # Assembly values: Y1, -Y1, -Y1, Y1, Y2, -Y2, -Y2, Y2
    y_vals = jnp.array([2.0, -2.0, -2.0, 2.0, 2.0, -2.0, -2.0, 2.0], dtype=jnp.complex128)
    
    r_idx = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    c_idx = np.array([0, 1, 0, 1, 1, 2, 1, 2])
    
    topology = NodalRepresentation(
        r_idx=r_idx, 
        c_idx=c_idx,
        ext_idx=np.array([0, 2]),
        int_idx=np.array([1])
    )
    
    result = solver.run(y_vals, topology)
    
    # Expected Equivalent Y-matrix for a 1.0 S series connection
    expected_y = jnp.array([
        [ 1.0, -1.0],
        [-1.0,  1.0]
    ], dtype=jnp.complex128)
    
    assert result.y.shape == (2, 2)
    np.testing.assert_allclose(result.y, expected_y, atol=1e-7)


def test_global_mna_reducer_aux_resistor():
    """
    Test Modified Nodal Analysis (MNA) using purely auxiliary variables.
    
    Models a 5 Ohm resistor (G = 0.2 S) between Node 0 and Node 1 using explicit 
    MNA branch equations rather than a standard Y matrix entry.
    
    Branch Equation: V_0 - V_1 - R * I_aux = 0
    """
    solver = GlobalMNAReducer()
    
    # Standard Y block is completely empty.
    y_vals = jnp.array([], dtype=jnp.complex128)
    y_r = np.array([], dtype=int)
    y_c = np.array([], dtype=int)
    
    # B Block (KCL Contributions of I_aux)
    # I_aux flows OUT of Node 0 (+1) and INTO Node 1 (-1)
    b_vals = jnp.array([1.0, -1.0], dtype=jnp.complex128)
    b_r = np.array([0, 1])
    b_c = np.array([0, 0])
    
    # C Block (Voltage Contributions to Branch Eq)
    # +1 * V_0 - 1 * V_1
    c_vals = jnp.array([1.0, -1.0], dtype=jnp.complex128)
    c_r = np.array([0, 0])
    c_c = np.array([0, 1])
    
    # D Block (Branch Equation Impedance)
    # -R * I_aux
    d_vals = jnp.array([-5.0], dtype=jnp.complex128)
    d_r = np.array([0])
    d_c = np.array([0])
    
    topology = MNARepresentation(
        y_r_idx=y_r, y_c_idx=y_c,
        b_r_idx=b_r, b_c_idx=b_c,
        c_r_idx=c_r, c_c_idx=c_c,
        d_r_idx=d_r, d_c_idx=d_c,
        ext_idx=np.array([0, 1]),
        int_idx=np.array([], dtype=int),
        aux_idx=np.array([0])
    )
    
    result = solver.run(y_vals, b_vals, c_vals, d_vals, topology)
    
    # The Schur complement should automatically condense the 3x3 MNA system 
    # back into the standard 2x2 Y-matrix of a 5-ohm resistor (Y = 1/5 = 0.2).
    expected_y = jnp.array([
        [ 0.2, -0.2],
        [-0.2,  0.2]
    ], dtype=jnp.complex128)
    
    assert result.y.shape == (2, 2)
    np.testing.assert_allclose(result.y, expected_y, atol=1e-7)


def test_global_mna_reducer_mixed_chain():
    """
    Test mixed MNA assembly with both Y-domain components and auxiliary components.
    
    Circuit:
    Node 0 (Ext) --[ Y1 = 2S ]-- Node 1 (Int) --[ MNA_R = 0.5 Ohms ]-- Node 2 (Ext)
    
    Total series resistance: 0.5 Ohms (Y1) + 0.5 Ohms (MNA_R) = 1.0 Ohms.
    Expected equivalent admittance is 1.0 S between Node 0 and Node 2.
    """
    solver = GlobalMNAReducer()
    
    # Y Block: 2.0 S admittance between Node 0 and Node 1
    y_vals = jnp.array([2.0, -2.0, -2.0, 2.0], dtype=jnp.complex128)
    y_r = np.array([0, 0, 1, 1])
    y_c = np.array([0, 1, 0, 1])
    
    # MNA Blocks: 0.5 Ohm Resistor between Node 1 and Node 2
    b_vals = jnp.array([1.0, -1.0], dtype=jnp.complex128)
    b_r = np.array([1, 2]) # Node 1, Node 2
    b_c = np.array([0, 0]) # Aux 0
    
    c_vals = jnp.array([1.0, -1.0], dtype=jnp.complex128)
    c_r = np.array([0, 0]) # Aux 0
    c_c = np.array([1, 2]) # Node 1, Node 2
    
    d_vals = jnp.array([-0.5], dtype=jnp.complex128)
    d_r = np.array([0])
    d_c = np.array([0])
    
    topology = MNARepresentation(
        y_r_idx=y_r, y_c_idx=y_c,
        b_r_idx=b_r, b_c_idx=b_c,
        c_r_idx=c_r, c_c_idx=c_c,
        d_r_idx=d_r, d_c_idx=d_c,
        ext_idx=np.array([0, 2]),
        int_idx=np.array([1]),
        aux_idx=np.array([0]) # K=1 auxiliary variable
    )
    
    result = solver.run(y_vals, b_vals, c_vals, d_vals, topology)
    
    expected_y = jnp.array([
        [ 1.0, -1.0],
        [-1.0,  1.0]
    ], dtype=jnp.complex128)
    
    assert result.y.shape == (2, 2)
    np.testing.assert_allclose(result.y, expected_y, atol=1e-7)