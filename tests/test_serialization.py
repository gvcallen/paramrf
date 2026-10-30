import json
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import pytest

import pmrf as prf
import pmrf.models as prm
from pmrf import serialization
from pmrf.materials import ConstantDielectric
from pmrf.materials.dielectric import AbstractDielectric
from pmrf.stats.distributions import Uniform
from tests._catalogue import LINES, MATERIALS


class DefaultsModule(eqx.Module):
    a: float
    formulation: str = "old"


def test_saved_defaults_survive_default_change(tmp_path, monkeypatch):
    path = tmp_path / "model.prf"
    prf.save(path, DefaultsModule(1.0))

    field = DefaultsModule.__dataclass_fields__["formulation"]
    monkeypatch.setattr(field, "default", "new")

    loaded = prf.load(path)
    assert loaded.a == 1.0
    assert loaded.formulation == "old"


def test_header_is_written(tmp_path):
    path = tmp_path / "model.prf"
    prf.save(path, DefaultsModule(1.0))

    data = json.loads(path.read_text())
    assert data["format"] == "prf"
    assert data["schema_version"] == serialization.SCHEMA_VERSION == 1
    assert data["paramrf_version"] is not None
    assert data["paramrf_version"] == prf.__version__
    assert data["tree"]["__class__"] == "DefaultsModule"
    assert data["tree"]["__state__"]["formulation"] == "old"


def test_missing_class_error_names_class_and_module(tmp_path):
    path = tmp_path / "model.prf"
    prf.save(path, DefaultsModule(1.0))

    data = json.loads(path.read_text())
    data["tree"]["__class__"] = "RenamedModule"
    data["tree"]["__module__"] = "downstream.models"
    path.write_text(json.dumps(data))

    with pytest.raises(ImportError) as info:
        prf.load(path)
    message = str(info.value)
    assert "RenamedModule" in message
    assert "downstream.models" in message
    assert "renamed or moved" in message


def test_headerless_file_is_rejected(tmp_path):
    path = tmp_path / "model.prf"
    path.write_text(json.dumps({"__type__": "__complex__", "__real__": 1.0, "__imag__": 0.0}))

    with pytest.raises(ValueError, match="header"):
        prf.load(path)


def test_newer_schema_version_is_rejected(tmp_path):
    path = tmp_path / "model.prf"
    prf.save(path, DefaultsModule(1.0))
    data = json.loads(path.read_text())
    data["schema_version"] = serialization.SCHEMA_VERSION + 1
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="schema_version"):
        prf.load(path)


# --- State is the real dataclass fields only --------------------------------

class _Strategy(eqx.Module):
    x: float


class ClassVarModule(eqx.Module):
    a: float
    strategy_type: ClassVar[type] = _Strategy
    flag: ClassVar[bool] = True


def _round_trip(tmp_path, tree):
    path = tmp_path / "model.prf"
    prf.save(path, tree)
    return prf.load(path)


def test_class_vars_are_neither_saved_nor_loaded(tmp_path):
    path = tmp_path / "model.prf"
    prf.save(path, ClassVarModule(1.0))

    state = json.loads(path.read_text())["tree"]["__state__"]
    assert set(state) == {"a"}

    loaded = prf.load(path)
    assert loaded.a == 1.0
    assert set(vars(loaded)) == {"a"}
    assert loaded.strategy_type is _Strategy
    assert loaded.flag is True


def test_dataclass_type_is_saved_by_reference(tmp_path):
    loaded = _round_trip(tmp_path, {"cls": _Strategy, "instance": _Strategy(2.0)})
    assert loaded["cls"] is _Strategy
    assert isinstance(loaded["instance"], _Strategy) and loaded["instance"].x == 2.0


FREQ = prf.Frequency(10, 500, 11, "MHz")


def test_microstrip_line_round_trips(tmp_path):
    line = prm.MicrostripLine(w=4e-3, h=1.5e-3, t=35e-6, length=0.13)
    loaded = _round_trip(tmp_path, line)
    assert jnp.allclose(loaded.s(FREQ), line.s(FREQ), rtol=0, atol=1e-12)


@pytest.mark.parametrize("line", LINES.values(), ids=LINES.keys())
def test_catalogued_line_round_trips(line, tmp_path):
    loaded = _round_trip(tmp_path, line)
    assert type(loaded) is type(line)
    assert jnp.allclose(loaded.s(FREQ), line.s(FREQ), rtol=0, atol=1e-12)


@pytest.mark.parametrize("material", MATERIALS.values(), ids=MATERIALS.keys())
def test_catalogued_material_round_trips(material, tmp_path):
    loaded = _round_trip(tmp_path, material)
    assert type(loaded) is type(material)
    expected, actual = material.properties(FREQ), loaded.properties(FREQ)
    names = ("ep_r", "mu_r") if isinstance(material, AbstractDielectric) else ("zs", "sigma")
    for name in names:
        assert jnp.allclose(getattr(actual, name), getattr(expected, name), rtol=0, atol=1e-12)


def test_cascade_with_free_line_parameters_round_trips(tmp_path):
    line = prm.MicrostripLine(
        w=prf.Random(Uniform(3e-3, 5e-3), value=4e-3),
        h=1.5e-3,
        t=35e-6,
        dielectric=ConstantDielectric(ep_r=prf.Random(Uniform(4.0, 4.6), value=4.3), tand=0.02),
        length=0.13,
    )
    cascade = (
        prm.Inductor(L=prf.Random(Uniform(1e-9, 1e-8), value=5e-9))
        ** line
        ** prm.ShuntCapacitor(C=1e-12)
    )
    loaded = _round_trip(tmp_path, cascade)

    expected = prf.values(cascade, free_only=True)
    actual = prf.values(loaded, free_only=True)
    assert len(expected) == 3
    assert actual.keys() == expected.keys()
    for name in expected:
        assert jnp.array_equal(actual[name], expected[name])
    assert jnp.allclose(loaded.s(FREQ), cascade.s(FREQ), rtol=0, atol=1e-12)


def test_class_var_keys_in_older_files_are_ignored(tmp_path):
    """0.38.0 wrote ClassVars such as `supports_native_z0` into the state."""
    path = tmp_path / "model.prf"
    prf.save(path, prm.Resistor(R=50.0))
    data = json.loads(path.read_text())
    data["tree"]["__state__"]["supports_native_z0"] = True
    path.write_text(json.dumps(data))

    loaded = prf.load(path)
    assert isinstance(loaded, prm.Resistor)
    assert "supports_native_z0" not in vars(loaded)
