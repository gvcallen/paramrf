import equinox as eqx
import pytest
from typing import Any

from pmrf.utils.optics import Lens, Traversal 

# ==========================================
# Test Fixtures & Models
# ==========================================

class Component(eqx.Module):
    name: str
    value: float
    is_active: bool = True

    def __post_init__(self):
        # A simple check to ensure Equinox validation runs during rebuilds
        if self.value < 0:
            raise ValueError("Component value cannot be negative.")

class System(eqx.Module):
    main_comp: Component
    backup_comp: Component
    comp_list: list[Component]
    comp_dict: dict[str, Component]
    
    @property
    def at(self) -> Lens["System"]:
        return Lens(self)

@pytest.fixture
def base_system() -> System:
    return System(
        main_comp=Component(name="Main", value=10.0),
        backup_comp=Component(name="Backup", value=5.0),
        comp_list=[
            Component(name="L1", value=1.0),
            Component(name="L2", value=2.0)
        ],
        comp_dict={
            "D1": Component(name="D1", value=100.0, is_active=False),
            "D2": Component(name="D2", value=200.0, is_active=True)
        }
    )

# ==========================================
# 1. Tests: Single-Target Lens
# ==========================================

def test_lens_get(base_system: System):
    """Test retrieving a value deep in the tree."""
    val = base_system.at.main_comp.value.get()
    assert val == 10.0

def test_lens_set_immutability(base_system: System):
    """Test setting an attribute doesn't mutate the original tree."""
    new_system = base_system.at.main_comp.value.set(99.0)
    
    assert new_system.main_comp.value == 99.0
    assert base_system.main_comp.value == 10.0 # Original is unchanged
    assert new_system.main_comp is not base_system.main_comp

def test_lens_getitem_index(base_system: System):
    """Test updating an item inside a list using indexing."""
    new_system = base_system.at.comp_list[0].value.apply(lambda x: x * 10)
    
    assert new_system.comp_list[0].value == 10.0
    assert new_system.comp_list[1].value == 2.0 # Other list item unchanged

def test_lens_getitem_string_attr(base_system: System):
    """Test using string keys in __getitem__ to access attributes."""
    attr_name = "backup_comp"
    new_system = base_system.at[attr_name].value.set(42.0)
    assert new_system.backup_comp.value == 42.0

def test_post_init_validation(base_system: System):
    """Ensure dataclasses.replace triggers __post_init__ during rebuild."""
    with pytest.raises(ValueError, match="Component value cannot be negative."):
        base_system.at.main_comp.value.set(-5.0)


# ==========================================
# 2. Tests: Multi-Target Traversal (Select)
# ==========================================

def test_traversal_select_get(base_system: System):
    """Test retrieving multiple targets simultaneously."""
    results = base_system.at.select("main_comp", "backup_comp").value.get()
    assert results == [10.0, 5.0]

def test_traversal_select_apply(base_system: System):
    """Test applying a function to multiple distinct attributes."""
    new_sys = base_system.at.select("main_comp", "backup_comp").value.apply(lambda x: x + 1.0)
    
    assert new_sys.main_comp.value == 11.0
    assert new_sys.backup_comp.value == 6.0


# ==========================================
# 3. Tests: Collections (Each)
# ==========================================

def test_traversal_each_list(base_system: System):
    """Test iterating over a list."""
    new_sys = base_system.at.comp_list.each().value.apply(lambda x: x * 2)
    
    assert new_sys.comp_list[0].value == 2.0
    assert new_sys.comp_list[1].value == 4.0

def test_traversal_each_dict(base_system: System):
    """Test iterating over dictionary values."""
    new_sys = base_system.at.comp_dict.each().value.set(0.0)
    
    assert new_sys.comp_dict["D1"].value == 0.0
    assert new_sys.comp_dict["D2"].value == 0.0

def test_each_type_error(base_system: System):
    """Ensure .each() throws an error on non-collections."""
    with pytest.raises(TypeError, match="Cannot iterate"):
        base_system.at.main_comp.each()


# ==========================================
# 4. Tests: Conditional Targeting (Filter)
# ==========================================

def test_lens_filter_attributes(base_system: System):
    """Test filtering attributes of a class (Lens.filter)."""
    # Freeze all Component instances at the root of System
    new_sys = base_system.at.filter(lambda x: isinstance(x, Component)).is_active.set(False)
    
    assert new_sys.main_comp.is_active is False
    assert new_sys.backup_comp.is_active is False
    
    # comp_list is a list, not a Component, so items inside shouldn't be affected
    assert new_sys.comp_list[0].is_active is True 

def test_traversal_filter_items(base_system: System):
    """Test filtering items within a collection (Traversal.filter)."""
    # Set value to 999 only for ACTIVE components in the dictionary
    new_sys = (
        base_system.at.comp_dict.each()
        .filter(lambda comp: comp.is_active)
        .value.set(999.0)
    )
    
    assert new_sys.comp_dict["D1"].value == 100.0 # Was inactive, unchanged
    assert new_sys.comp_dict["D2"].value == 999.0 # Was active, updated


# ==========================================
# 5. Tests: Deep Nested Chaining
# ==========================================

def test_complex_deep_chaining(base_system: System):
    """Test combining each, filter, select, and deep attribute access."""
    
    def double_if_active(sys: System) -> System:
        return (
            sys.at.comp_list.each()
            .filter(lambda c: c.is_active)
            .value.apply(lambda x: x * 2)
        )
    
    new_sys = double_if_active(base_system)
    assert new_sys.comp_list[0].value == 2.0
    assert new_sys.comp_list[1].value == 4.0