# tests/test_fitting/test_routers.py

import pytest
import numpy as np
import jax.numpy as jnp

from pmrf.models import Model
from pmrf.parameters import Bounded, Fixed
from pmrf.frequency import Frequency
from pmrf.network_collection import NetworkCollection
from pmrf.fitting.routers import fit_sequential, fit_joint

# ==========================================
# 1. Dummy Models & Objectives
# ==========================================

class SubModel(Model):
    val: object
    
    def s(self, freq: Frequency):
        # A simple scalar S-parameter response to fit against
        return jnp.ones((freq.npoints, 1, 1), dtype=complex) * self.val

class CompositeModel(Model):
    sub1: SubModel
    sub2: SubModel
    global_fixed: object = Fixed(10.0)

@pytest.fixture
def freq():
    return Frequency(start=1.0, stop=2.0, npoints=2, unit='GHz')

@pytest.fixture
def starting_model():
    return CompositeModel(
        sub1=SubModel(val=Bounded(0.0, 10.0, value=0.0)),
        sub2=SubModel(val=Bounded(0.0, 10.0, value=0.0))
    )

@pytest.fixture
def target_collection(freq):
    skrf = pytest.importorskip("skrf")
    
    # Sub-network 1 targets sub1.val = 3.0
    n1 = skrf.Network(frequency=freq.to_skrf(), s=np.ones((2, 1, 1)) * 3.0, name='sub1')
    # Sub-network 2 targets sub2.val = 7.0
    n2 = skrf.Network(frequency=freq.to_skrf(), s=np.ones((2, 1, 1)) * 7.0, name='sub2')
    
    return NetworkCollection([n1, n2])

# ==========================================
# 2. Router Tests
# ==========================================

def test_fit_sequential(starting_model, target_collection):
    """
    Tests that fit_sequential correctly isolates sub-models, fits them,
    and sequentially recombines them without affecting earlier parameters
    or global fixed parameters.
    """
    from pmrf.optimize import ScipyMinimize
    solver = ScipyMinimize()
    
    final_model, results_dict = fit_sequential(
        model=starting_model,
        data=target_collection,
        solver=solver
    )
    
    # Verify sub-models were correctly fitted to their respective targets
    assert jnp.allclose(final_model.sub1.val.value, 3.0, atol=1e-3)
    assert jnp.allclose(final_model.sub2.val.value, 7.0, atol=1e-3)
    
    # Verify the main model's fixed parameter remains untouched
    assert final_model.global_fixed.value == 10.0
    
    # Verify the localized result dictionaries are successfully populated
    assert 'sub1' in results_dict
    assert 'sub2' in results_dict


# def test_fit_joint(starting_model, target_collection):
#     """
#     Tests that fit_joint successfully builds a joint target graph across
#     all networks in the collection and fits them simultaneously.
#     """
#     from pmrf.optimize import ScipyMinimize
#     solver = ScipyMinimize()
    
#     result = fit_joint(
#         model=starting_model,
#         data=target_collection,
#         solver=solver
#     )
    
#     final_model = result.model
    
#     # Verify both sub-models converged simultaneously
#     assert jnp.allclose(final_model.sub1.val.value, 3.0, atol=1e-3)
#     assert jnp.allclose(final_model.sub2.val.value, 7.0, atol=1e-3)
    
#     # Verify global parameter preservation
#     assert final_model.global_fixed.value == 10.0