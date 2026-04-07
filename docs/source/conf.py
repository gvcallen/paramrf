# docs/source/conf.py
import types
import logging

class MuteMathDollarWarnings(logging.Filter):
    """Mute aggressive node warnings from sphinx-math-dollar."""
    def filter(self, record):
        return 'pending_xref_condition' not in record.getMessage()

# Attach the filter to the root logger
logging.getLogger().addFilter(MuteMathDollarWarnings())

import os
import sys
from pathlib import Path

import sphinx.addnodes
from sphinx_math_dollar import NODE_BLACKLIST

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

    try:
        from importlib.metadata import version as _dist_version
        return _dist_version(_project_name)
    except Exception:
        pass

    return "0.0.0"

# --- Project info ------------------------------------------------------------
project = 'paramrf'
author = 'Gary Allen'
release = _get_release(project, _repo_root)
version = ".".join(release.split(".")[:2]) if release and "." in release else release

# --- Extensions & config -----------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.doctest',       # Required for NumPyro example blocks
    'sphinx_math_dollar',
    'sphinx.ext.mathjax',       # Required for rendering LaTeX equations
    'sphinx.ext.intersphinx',   # Links to external docs (JAX, NumPy, etc.)
    'myst_parser',              # For Markdown pages
]

# --- Autosummary & Autodoc Tuning ---
autosummary_generate = True
autosummary_ignore_module_all = False  # Forces Sphinx to respect your __all__ lists
add_module_names = False               # Strips the fully qualified paths from signatures
autoclass_content = 'class'

# Type hinting formatting (crucial for JAX arrays)
autodoc_typehints = 'description'
python_use_unqualified_type_names = True 

# Stop Python from copying parent docstrings into children automatically
autodoc_inherit_docstrings = True 

# Unwrap NumPyro aliases so they document fully instead of saying "alias of..."
autodoc_type_aliases = {
    'UniformDistribution': 'numpyro.distributions.Uniform',
    'LogUniformDistribution': 'numpyro.distributions.LogUniform',
    'NormalDistribution': 'numpyro.distributions.Normal',
    'MultivariateNormalDistribution': 'numpyro.distributions.MultivariateNormal',
    'LogNormalDistribution': 'numpyro.distributions.LogNormal',
}

autodoc_member_order = 'bysource'
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "inherited-members": True,    
    "show-inheritance": True,
    "imported-members": True,     
    "member-order": "bysource",
    "exclude-members": "__weakref__"
}

# --- Intersphinx Configuration ---
# Creates clickable links to standard scientific Python libraries
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}

# --- Napoleon (NumPy Docstring) Settings ---
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

# --- MyST Markdown Settings ---
myst_enable_extensions = ['colon_fence', 'deflist', 'linkify', 'dollarmath']
myst_heading_anchors = 3

# --- Math Dollar Configuration ---
# Silence the pending_xref_condition warnings
math_dollar_node_blacklist = NODE_BLACKLIST + (sphinx.addnodes.pending_xref_condition,)

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)'], ['$', '$']],
    "displayMath": [["\\[", "\\]"], ["$$", "$$"]],
  }
}

# --- HTML & Theme Settings ---
templates_path = ['_templates']
exclude_patterns = []
html_theme = 'sphinx_rtd_theme'
html_static_path = []  # Reverted back to an empty list

# --- Event Hooks ---
def skip_member(app, what, name, obj, skip, options):
    """Custom logic for skipping specific documentation members."""
    
    # 1. Skip members marked with the internal auto flag
    if what == "class" and getattr(obj, "_pmrf_auto", False):
        return True

    # 2. Prevent sub-modules from being documented as members of their parent namespace
    if isinstance(obj, types.ModuleType):
        return True

    # 3. Filter external inherited members from `parax`
    # First, figure out where the object was originally defined
    obj_module = getattr(obj, '__module__', None)
    
    # Python properties don't hold __module__ directly, so we check their getter
    if obj_module is None and isinstance(obj, property):
        obj_module = getattr(obj.fget, '__module__', '')

    # If the method/attribute originated from the 'parax' package...
    if obj_module and obj_module.startswith('parax'):
        # 'name' is the full path Sphinx is documenting (e.g., 'pmrf.core.Model.tree_flatten')
        # If we are documenting the base Model class, allow the inherited methods
        if '.Model.' in name:
            return skip 
            
        # For all other subclasses (Resistor, etc.), hide the parax methods
        return True

    return skip

def process_docstring(app, what, name, obj, options, lines):
    """Prevent base Model class docstrings from bleeding into component subclasses."""
    if what == "class" and isinstance(obj, type):
        try:
            from pmrf.core import Model
            # If it is a subclass of Model, but NOT the Model base class itself
            if issubclass(obj, Model) and obj is not Model:
                # If the child's docstring is perfectly identical to the parent's, 
                # Python inherited it automatically. Wipe it clean.
                if getattr(obj, '__doc__', None) == getattr(Model, '__doc__', None):
                    lines[:] = []  
        except ImportError:
            pass

def setup(app):
    app.connect("autodoc-skip-member", skip_member)
    app.connect("autodoc-process-docstring", process_docstring)