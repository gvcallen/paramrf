import json

import equinox as eqx
import pytest

import pmrf as prf
from pmrf import serialization


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
