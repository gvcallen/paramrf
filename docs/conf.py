# docs/source/conf.py
import os
import sys
import types
from importlib.metadata import version as get_version, PackageNotFoundError

import sphinx.addnodes
from sphinx_math_dollar import NODE_BLACKLIST

# --- 1. Path Setup -----------------------------------------------------------
_repo_root = os.path.abspath('../')
sys.path.insert(0, os.path.join(_repo_root, 'src'))
sys.path.insert(0, _repo_root)

# --- 2. Project Info ---------------------------------------------------------
project = 'paramrf'
author = 'Gary Allen'

try:
    release = get_version(project)
    version = ".".join(release.split(".")[:2]) if "." in release else release
except PackageNotFoundError:
    release = version = "0.0.0"

# --- 3. Extensions -----------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx_math_dollar',      # Parses $math$ in .rst files
    'sphinx.ext.mathjax',      # Renders the math via MathJax
    'myst_parser',             # Parses markdown files
]

# --- 4. Sphinx Options -------------------------------------------------------
nbsphinx_execute = 'auto'
autoclass_content = 'both'
autosummary_generate = True
autosummary_ignore_module_all = False
templates_path = ['_templates']
exclude_patterns = ['_templates', '_build', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    'inherited-members': False,
    "member-order": "groupwise", 
    "special-members": "__call__,__getitem__,__len__,__add__,__sub__,__mul__,__rmul__,__div__,__truediv__,__floordiv__,__mod__", 
}

autodoc_type_aliases = {
    'Param': 'Param',
}

# --- 5. Napoleon Configuration -----------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True
napoleon_use_param = True

# --- 6. Math & MyST Configuration --------------------------------------------
myst_enable_extensions = ['colon_fence', 'deflist', 'linkify', 'dollarmath']

# Silences the aggressive warnings from sphinx-math-dollar
math_dollar_node_blacklist = NODE_BLACKLIST + (sphinx.addnodes.pending_xref_condition,)

# Configures MathJax to recognize your dollar sign syntax
mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)'], ['$', '$']],
    "displayMath": [["\\[", "\\]"], ["$$", "$$"]],
  }
}

# --- 7. HTML Theme -----------------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# --- 8. Event Hooks ----------------------------------------------------------
def is_pmrf_auto(obj):
    """Robustly detect auto-generated pmrf methods."""
    if getattr(obj, "_pmrf_auto", False):
        return True
    
    # Bound method → underlying function
    func = getattr(obj, "__func__", None)
    if func and getattr(func, "_pmrf_auto", False):
        return True

    # Property → getter function
    if isinstance(obj, property):
        if getattr(obj.fget, "_pmrf_auto", False):
            return True

    return False


def skip_member(app, what, name, obj, skip, options):
    # 1. Skip pmrf auto-generated methods
    if is_pmrf_auto(obj):
        return True

    # 2. Skip modules
    if isinstance(obj, types.ModuleType):
        return True

    # 3. Skip parax stuff
    obj_module = getattr(obj, '__module__', None)
    if obj_module is None and isinstance(obj, property):
        obj_module = getattr(obj.fget, '__module__', '')

    # 4. Skip specific single-letter methods/properties globally
    methods_to_hide = {"s", "y", "z", "a", "build"}
    if name in methods_to_hide:
        return True

    return skip

def setup(app):
    # Only connect the single unified function
    app.connect("autodoc-skip-member", skip_member)