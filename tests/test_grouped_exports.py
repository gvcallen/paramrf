"""Grouped export checks. Run with ``python -m pytest tests/test_grouped_exports.py``.

The static checks use the test extra's pinned Jedi 0.20.0. They check source
analysis, not any particular notebook editor or language server.
"""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import jedi
import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("objectives", "stats")


def exports(package):
    source = ast.parse((ROOT / "pmrf" / package / "__init__.py").read_text())
    assignment = next(
        node for node in source.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_EXPORTS"
                for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


def infer(source):
    line = len(source.splitlines())
    return jedi.Script(source, project=jedi.Project(str(ROOT))).infer(
        line=line, column=len(source.splitlines()[-1])
    )


@pytest.mark.parametrize("package", PACKAGES)
def test_jedi_resolves_every_declared_export(package):
    unresolved = [
        name for name in exports(package)
        if not infer(f"import pmrf as prf\nprf.{package}.{name}")
    ]
    assert not unresolved


@pytest.mark.parametrize("package,name", [("objectives", "Goal"), ("stats", "Normal")])
def test_jedi_resolves_direct_and_qualified_imports(package, name):
    for source in (
        f"import pmrf as prf\nprf.{package}.{name}",
        f"import pmrf.{package} as grouped\ngrouped.{name}",
        f"from pmrf import {package}\n{package}.{name}",
        f"from pmrf.{package} import {name}\n{name}",
    ):
        assert infer(source), source


@pytest.mark.parametrize("package,name", [("objectives", "Goal"), ("stats", "Normal")])
def test_jedi_completes_grouped_names(package, name):
    source = f"import pmrf as prf\nprf.{package}."
    completions = jedi.Script(source, project=jedi.Project(str(ROOT))).complete(
        line=2, column=len(source.splitlines()[-1])
    )
    assert name in {completion.name for completion in completions}


def test_jedi_infers_goal_instance():
    result = infer(
        "import pmrf as prf\n"
        "goal = prf.objectives.Goal('s11_db', '<', -20)\n"
        "goal"
    )
    assert any(item.name == "Goal" and item.type == "instance" for item in result)


@pytest.mark.parametrize("package", PACKAGES)
def test_runtime_exports_keep_implementation_identity(package):
    grouped = importlib.import_module(f"pmrf.{package}")
    import pmrf as prf

    assert set(exports(package)).issubset(grouped.__all__)
    for name in set(grouped.__all__) - set(exports(package)):
        implementation = importlib.import_module(f"pmrf.{package}.{name}")
        assert getattr(grouped, name) is implementation
        assert getattr(getattr(prf, package), name) is implementation

    for name, module in exports(package).items():
        implementation = getattr(importlib.import_module(f"pmrf.{package}.{module}"), name)
        assert getattr(grouped, name) is implementation
        assert getattr(getattr(prf, package), name) is implementation
        assert getattr(importlib.import_module(f"pmrf.{package}"), name) is implementation
        namespace = {}
        exec(f"from pmrf.{package} import {name}", namespace)
        assert namespace[name] is implementation

    with pytest.raises(AttributeError):
        getattr(grouped, "NotAnExport")


@pytest.mark.parametrize("package", PACKAGES)
def test_grouped_initializer_keeps_implementations_lazy(package):
    script = f"""
import sys
import types
root = types.ModuleType('pmrf')
root.__path__ = [{str(ROOT / 'pmrf')!r}]
sys.modules['pmrf'] = root
import pmrf.{package}
loaded = {{name for name in sys.modules if name.startswith('pmrf.{package}.')}}
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-P", "-c", script], cwd=ROOT, check=True)


@pytest.mark.parametrize("import_statement", [
    "import pmrf",
    "import pmrf.objectives",
    "import pmrf.stats",
])
def test_fresh_import_does_not_load_every_grouped_implementation(import_statement):
    modules = {
        package: sorted(set(exports(package).values())) for package in PACKAGES
    }
    script = f"""
import sys
{import_statement}
for package, modules in {modules!r}.items():
    assert not all(f'pmrf.{{package}}.{{module}}' in sys.modules for module in modules), package
"""
    subprocess.run([sys.executable, "-P", "-c", script], cwd=ROOT, check=True)


def test_root_lazy_packages_remain_available():
    script = """
import importlib
import pmrf
for name in ('fitting', 'infer', 'optimize', 'viz'):
    assert name in dir(pmrf)
    assert getattr(pmrf, name) is importlib.import_module(f'pmrf.{name}')
"""
    subprocess.run([sys.executable, "-P", "-c", script], cwd=ROOT, check=True)


@pytest.mark.parametrize("package,name,module", [
    ("objectives", "Goal", "evaluators"),
    ("stats", "AutoCrossNoise", "noise_models"),
])
def test_fresh_export_access_loads_defining_module(package, name, module):
    script = f"""
import sys
import pmrf as prf
assert getattr(prf.{package}, '{name}') is not None
assert 'pmrf.{package}.{module}' in sys.modules
"""
    subprocess.run([sys.executable, "-P", "-c", script], cwd=ROOT, check=True)
