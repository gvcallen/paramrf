import pytest
import pmrf as prf
from pmrf.models import Resistor, Inductor, Capacitor, Cascade

def test_explicit_parameter_name():
    """Test that directly naming a parameter overrides the JAX path."""
    val = prf.Unconstrained(2.0, name="custom_param")
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


def test_at_string_target():
    """Test that .at() accepts a string parameter name."""
    r = Resistor(prf.Unconstrained(50.0, name="custom_R"))
    
    val = r.at("custom_R").get()
    assert prf.unwrap(val) == 50.0
    
    new_r = r.at("custom_R").set(prf.Unconstrained(100.0, name="custom_R"))
    assert prf.unwrap(new_r.at("custom_R").get()) == 100.0

def test_at_multiple_string_targets():
    """Test that .at() accepts an iterable of string parameter names."""
    rc = Resistor(prf.Unconstrained(50.0, name="custom_R")) ** Capacitor(prf.Unconstrained(10.0, name="custom_C"))
    
    # Get multiple using a tuple
    vals = rc.at(("custom_R", "custom_C")).get()
    unwrapped_vals = tuple(prf.unwrap(v) for v in vals)
    assert unwrapped_vals == (50.0, 10.0)
    
    # Set multiple using a list
    new_rc = rc.at(["custom_R", "custom_C"]).set((
        prf.Unconstrained(100.0, name="custom_R"), 
        prf.Unconstrained(20.0, name="custom_C")
    ))
    assert prf.unwrap(new_rc.at("custom_R").get()) == 100.0
    assert prf.unwrap(new_rc.at("custom_C").get()) == 20.0

def test_tied_string_targets():
    """Test that .tied() accepts string parameter names for source and target."""
    from pmrf.models import Wrapped
    from pmrf.modules import Tied
    
    rc = Resistor(prf.Unconstrained(50.0, name="custom_R")) ** Capacitor(prf.Unconstrained(10.0, name="custom_C"))
    
    # Tie custom_R to custom_C using strings
    tied_rc = rc.tied(target="custom_R", source="custom_C", tie_fn=lambda c: c * 5.0)
    
    # If the resolution failed, it would throw an error before instantiation
    assert isinstance(tied_rc, Wrapped)
    assert isinstance(tied_rc.wrapped, Tied)

def test_target_resolution_errors():
    """Test that invalid target formats or non-existent names raise appropriate errors."""
    r = Resistor(prf.Unconstrained(50.0, name="custom_R"))
    
    # Test non-existent string name
    with pytest.raises(ValueError, match="not resolve parameter name"):
        r.at("nonexistent_param")
        
    with pytest.raises(ValueError, match="not resolve parameter name"):
        r.at(123)
        
    # Test that tied checks both target and source
    with pytest.raises(ValueError, match="not found in the provided lookup"):
        r.tied(target="custom_R", source="nonexistent_param")

def test_at_nested_namespace():
    """Test that .at() resolves string targets using nested model namespaces."""
    r = Resistor(prf.Unconstrained(50.0, name="res_val"), name="myR")
    cas1 = Cascade([r], name="myCas")
    
    # Go a level deeper: cas2 acts as the root, so cas1's name ("myCas") 
    # will be properly traversed and added to the namespace.
    cas2 = Cascade([cas1])
    
    expected_namespace_name = "myCas_myR_res_val"
    
    # Verify the value can be retrieved using the fully namespaced string
    val = cas2.at(expected_namespace_name).get()
    assert prf.unwrap(val) == 50.0
    
    # Verify the value can be updated using the fully namespaced string
    new_cas2 = cas2.at(expected_namespace_name).set(prf.Unconstrained(100.0, name="res_val"))
    assert prf.unwrap(new_cas2.at(expected_namespace_name).get()) == 100.0


def test_tied_nested_namespace():
    """Test that .tied() resolves string targets using nested model namespaces."""
    from pmrf.models import Wrapped
    from pmrf.modules import Tied
    
    r = Resistor(prf.Unconstrained(50.0, name="res_val"), name="myR")
    c = Capacitor(prf.Unconstrained(10.0, name="cap_val"), name="myC")
    cas1 = Cascade([r, c], name="myCas")
    
    # Go a level deeper so "myCas" acts as a namespace prefix for its children
    cas2 = Cascade([cas1])
    
    target_name = "myCas_myR_res_val"
    source_name = "myCas_myC_cap_val"
    
    # Tie the nested resistor's value to the nested capacitor's value
    tied_cas = cas2.tied(
        target=target_name,
        source=source_name,
        tie_fn=lambda val: val * 5.0
    )
    
    assert isinstance(tied_cas, Wrapped)
    assert isinstance(tied_cas.wrapped, Tied)
