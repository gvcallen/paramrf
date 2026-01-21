# docs/source/conf.py
import os
import sys
from pathlib import Path

# Compute repo roots relative to this conf.py file
_here = os.path.abspath(os.path.dirname(__file__))          # docs/source
_repo_root = os.path.abspath(os.path.join(_here, '..', '..'))  # repo_root
_src_root = os.path.join(_repo_root, 'src')                 # repo_root/src

for p in (_src_root, _repo_root):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# --- Version helpers ---------------------------------------------------------
def _get_release(_project_name: str, repo_root: str) -> str:
    """Resolve package version for Sphinx 'release' from multiple sources."""
    # 1) Try pyproject.toml (PEP 621)
    try:
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # fallback for older Pythons

        pyproject = Path(repo_root) / "pyproject.toml"
        if pyproject.is_file():
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            ver = (data.get("project") or {}).get("version")
            if ver:
                return ver
    except Exception:
        pass

    # 2) If the package is installed in the env, ask importlib.metadata
    try:
        from importlib.metadata import version as _dist_version
        return _dist_version(_project_name)
    except Exception:
        pass

    # 3) Final fallback
    return "0.0.0"

# --- Project info ------------------------------------------------------------
project = 'paramrf'
author = 'Gary Allen'
release = _get_release(project, _repo_root)
# Sphinx often uses 'version' as the short X.Y string
version = ".".join(release.split(".")[:2]) if release and "." in release else release

# --- Extensions & config -----------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',  # keep if you have any .md pages
]

autosummary_generate = True
autoclass_content = 'class'
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": "__init__",
    "inherited-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "exclude-members": "__weakref__"
}

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = False

myst_enable_extensions = ['colon_fence', 'deflist', 'linkify']
myst_heading_anchors = 3

templates_path = ['_templates']
exclude_patterns = []
html_theme = 'sphinx_rtd_theme'
html_static_path = []

def skip_member(app, what, name, obj, skip, options):
    if what == "class" and getattr(obj, "_pmrf_auto", False):
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_member)