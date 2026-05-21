import pytest
import pmrf as prf
from pmrf.models import Resistor, Inductor, Capacitor, Cascade

def test_explicit_parameter_name():
    """Test that directly naming a parameter overrides the JAX path."""
    val = prf.Variable(2.0, name="custom_param")
    res = Resistor(val)
    params = res.named_params()
    
    assert "custom_param" in params
    assert params["custom_param"] == 2.0

def test_model_namespace():
    """Test that naming a model prefixes its child parameters when contained in another model."""
    res = Resistor(2.0, name="myR")
    cas1 = Cascade([res], name="myCas")
    cas2 = Cascade([cas1])
    params = cas2.named_params()
    
    assert any(k.startswith("myCas_myR") for k in params.keys())

def test_custom_namespace_separator():
    """Test the namespace_separator argument."""
    res = Resistor(2.0, name="myR")
    cas1 = Cascade([res], name="myCas")
    cas2 = Cascade([cas1])
    params = cas2.named_params(namespace_separator='*')
    
    assert any(k.startswith("myCas*myR") for k in params.keys())

def test_name_collision_raises_error():
    """Test that identical names in the same hierarchy raise a ValueError."""
    res = Resistor(2.0, name="myR")
    ind = Inductor(2.0)
    cap = Capacitor(2.0)

    cas1 = Cascade([res, ind, cap])
    cas2 = Cascade([res, ind, cap])
    
    combined_model = cas1 ** cas2

    with pytest.raises(ValueError, match="name collision"):
        combined_model.named_params()