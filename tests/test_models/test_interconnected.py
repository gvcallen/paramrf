# tests/test_models/test_interconnected.py
import pytest
import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.models import Resistor, Short, Port, Circuit, Cascade, Terminated, Capacitor, Inductor

@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')

# ---------------------------------------------------------
# Circuit Tests
# ---------------------------------------------------------

def test_circuit_valid_structure(basic_freq):
    """Test a valid circuit connection (simple series resistor)."""
    R = Resistor(R=50.0)
    p0 = Port()
    p1 = Port()
    
    connections = [
        [(p0, 0), (R, 0)],
        [(p1, 0), (R, 1)]
    ]
    
    circ = Circuit(connections)
    s = circ.s(basic_freq)
    
    # Exposes 2 ports
    assert s.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s))

def test_circuit_invalid_collection_type():
    """Ensure passing a non-list as the root connections fails."""
    R = Resistor(50.0)
    p0 = Port()
    
    # Passing a tuple instead of a list of lists
    connections = ( [(p0, 0), (R, 0)], )
    with pytest.raises(TypeError, match="`connections` must be a list of lists"):
        Circuit(connections)

def test_circuit_invalid_node_type():
    """Ensure passing a non-list as a node fails."""
    R = Resistor(50.0)
    p0 = Port()
    
    # Passing a tuple for the inner node
    connections = [ ((p0, 0), (R, 0)) ]
    with pytest.raises(TypeError, match="must be a list of \\(Model, port_index\\)"):
        Circuit(connections)

def test_circuit_invalid_item_type():
    """Ensure passing malformed item tuples fails."""
    R = Resistor(50.0)
    p0 = Port()
    
    # Passing a 3-element tuple instead of 2
    connections = [ [(p0, 0, 'extra_arg'), (R, 0)] ]
    with pytest.raises(TypeError, match="Must be a tuple of \\(Model, port_index\\)"):
        Circuit(connections)

def test_circuit_port_out_of_bounds():
    """Ensure connecting to an invalid port index fails."""
    R = Resistor(50.0)
    p0 = Port()
    
    connections = [ [(p0, 0), (R, 2)] ] # Index 2 is out of bounds
    with pytest.raises(ValueError, match="out of bounds"):
        Circuit(connections)

def test_circuit_duplicate_port():
    """Ensure a specific model's port cannot be connected to multiple nodes."""
    R = Resistor(50.0)
    p0 = Port()
    p1 = Port()
    
    connections = [
        [(p0, 0), (R, 0)],
        [(p1, 0), (R, 0)] # Cannot connect R's port 0 to two different nodes
    ]
    with pytest.raises(ValueError, match="is connected multiple times"):
        Circuit(connections)

# ---------------------------------------------------------
# Cascade Tests
# ---------------------------------------------------------

def test_cascade_valid(basic_freq):
    """Test standard valid cascading of 2-port networks."""
    R1 = Resistor(50.0)
    R2 = Resistor(100.0)
    
    cascaded = Cascade([R1, R2])
    s = cascaded.s(basic_freq)
    
    assert s.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s))

def test_cascade_one(basic_freq):
    """Test that a cascade of a single model remains the same."""
    R1 = Resistor(100.0)
    
    cascaded = Cascade([R1])
    s = cascaded.s(basic_freq)

    assert jnp.allclose(s, R1.s(basic_freq))

def test_cascade_invalid_odd_ports():
    """Cascade requires 2N-port networks."""
    R1 = Resistor(50.0) # 2-port
    S1 = Short()        # 1-port
    
    with pytest.raises(ValueError, match="All networks must be 2N-ports for Cascade"):
        Cascade([R1, S1])

def test_cascade_flattening():
    """Ensure nested cascades are flattened correctly."""
    R1 = Resistor(10.0)
    R2 = Resistor(20.0)
    R3 = Resistor(30.0)
    
    nested = Cascade([R1, Cascade([R2, R3])])
    
    # The models tuple should be flattened to just (R1, R2, R3)
    assert len(nested.cascade) == 3
    assert nested.cascade[0] is R1
    assert nested.cascade[2] is R3
    

# ---------------------------------------------------------
# Terminated Tests
# ---------------------------------------------------------

def test_terminated_valid(basic_freq):
    """Test terminating a 2-port network with a 1-port load."""
    R = Resistor(50.0) # 2-port
    S = Short()        # 1-port
    
    term = Terminated(R, S)
    s = term.s(basic_freq)
    
    # 2-port terminated in 1-port yields a 1-port network
    assert s.shape == (5, 1, 1)
    assert not jnp.any(jnp.isnan(s))

def test_terminated_invalid_port_ratios():
    """Test terminating a network with mismatched port ratios."""
    R1 = Resistor(50.0) # 2-port
    R2 = Resistor(50.0) # 2-port
    
    # Cannot terminate a 2-port in a 2-port directly with this class
    with pytest.raises(ValueError, match="Terminated only supports terminating 2N port networks in a 1N port"):
        Terminated(terminated_from=R1, terminated_into=R2)


def test_circuit_flatten_hierarchy(basic_freq):
    """
    Ensure nested Circuits and Cascades are perfectly flattened into a 
    single-level netlist of base components, dropping virtual interfaces.
    """
    # Base leaf components
    R1 = Resistor(10.0)
    R2 = Resistor(20.0)
    C1 = Capacitor(1e-12)
    L1 = Inductor(1e-9)

    # Create a Sub-circuit (R1 and R2 in series)
    sub_p0 = Port()
    sub_p1 = Port()
    sub_connections = [
        [(sub_p0, 0), (R1, 0)],
        [(R1, 1), (R2, 0)],
        [(sub_p1, 0), (R2, 1)]
    ]
    sub_circ = Circuit(sub_connections) # 2-port sub-circuit

    # Create a Sub-cascade (C1 and L1 in series)
    # Note: we disable automatic flatten on init so the graph algorithm has to do it.
    sub_casc = Cascade([C1, L1], flatten=False) # 2-port cascade

    # Top-level circuit connecting sub_circ -> sub_casc
    top_p0 = Port()
    top_p1 = Port()
    top_connections = [
        [(top_p0, 0), (sub_circ, 0)],
        [(sub_circ, 1), (sub_casc, 0)],
        [(sub_casc, 1), (top_p1, 0)]
    ]
    top_circ = Circuit(top_connections)

    # Evaluate S-parameters (this internally triggers self.flattened)
    s = top_circ.s(basic_freq)

    # Check that the numerical solve succeeds and exposes exactly 2 ports
    assert s.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s))

    # --- Verify the flattened graph topography ---
    flat_circ = top_circ.flattened
    flat_models = flat_circ.circuit

    # The flattened circuit should contain NO Circuit or Cascade wrapper objects.
    for m in flat_models:
        assert not isinstance(m, (Circuit, Cascade)), f"Flattening failed, found wrapper container: {type(m)}"

    # It should contain exactly 6 leaf items: the 2 top-level ports + 4 internal components
    assert len(flat_models) == 6
    
    # Verify the exact objects survived
    assert top_p0 in flat_models
    assert top_p1 in flat_models
    assert R1 in flat_models
    assert R2 in flat_models
    assert C1 in flat_models
    assert L1 in flat_models