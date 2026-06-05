# docs/source/conf.py
import os
import sys
import types
from importlib.metadata import version as get_version, PackageNotFoundError
import pkgutil
import importlib
import inspect
import sphinx.addnodes
from sphinx_math_dollar import NODE_BLACKLIST

def generate_models_tree(app):
    """Dynamically generates a nested bulleted list of the pmrf.models hierarchy 
    and a hidden autosummary block to generate the stubs."""
    import pmrf.models
    
    all_stubs = []
    
    def get_all_classes_flat(module):
        """Recursively scrapes all classes from a module and its subdirectories."""
        classes = []
        # 1. Grab classes in the current file
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if getattr(obj, '__module__', None) == module.__name__:
                classes.append((name, module.__name__))
                
        # 2. Walk subdirectories and grab their classes too
        if hasattr(module, '__path__'):
            for module_info in pkgutil.iter_modules(module.__path__):
                sub_name = f"{module.__name__}.{module_info.name}"
                try:
                    sub = importlib.import_module(sub_name)
                    classes.extend(get_all_classes_flat(sub))
                except ImportError:
                    pass
        return classes

    def walk_module(module, depth=0):
        lines = []
        indent = "    " * depth
        
        # 1. Document classes directly inside this module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if getattr(obj, '__module__', None) == module.__name__:
                lines.append(f"{indent}* :class:`~{module.__name__}.{name}`")
                all_stubs.append(f"{module.__name__}.{name}")
        
        # 2. Recursively find and document submodules
        if hasattr(module, '__path__'):
            for module_info in pkgutil.iter_modules(module.__path__):
                submodule_name = f"{module.__name__}.{module_info.name}"
                try:
                    submodule = importlib.import_module(submodule_name)
                    
                    if getattr(submodule, '__sphinx_group__', False):
                        # Add the grouped header as a main bullet
                        lines.append(f"{indent}* **{module_info.name}** (:mod:`~{submodule_name}`)")
                        
                        # Grab all nested classes from the directory tree and flatten them
                        flat_classes = get_all_classes_flat(submodule)
                        
                        # --- Sort alphabetically by class name ---
                        flat_classes.sort(key=lambda x: x[0])
                        
                        # Indent them exactly one level under the grouped header
                        child_indent = indent + "    "
                        for cls_name, cls_mod in flat_classes:
                            lines.append(f"{child_indent}* :class:`~{cls_mod}.{cls_name}`")
                            # Continue to queue them up for individual autosummary pages
                            all_stubs.append(f"{cls_mod}.{cls_name}")
                            
                    else:
                        # Add the module name as a header/bullet
                        lines.append(f"{indent}* **{module_info.name}** (:mod:`~{submodule_name}`)")
                        # Recurse deeper normally
                        lines.extend(walk_module(submodule, depth + 1))
                except ImportError:
                    pass
                    
        return lines

    # Prepend the page header
    tree_lines = [
        "Model Hierarchy", 
        "===============", 
        ""
    ]
    tree_lines.extend(walk_module(pmrf.models))
    
    # Append a HIDDEN autosummary block to force page generation
    tree_lines.extend([
        "",
        ".. raw:: html",
        "",
        "   <div style=\"display: none;\">",
        "",
        ".. autosummary::",
        "   :toctree: generated/",
        ""
    ])
    for stub in all_stubs:
        tree_lines.append(f"   {stub}")
    tree_lines.extend([
        "",
        ".. raw:: html",
        "",
        "   </div>",
        ""
    ])
    
    # Save directly to the models directory
    output_path = os.path.join(app.srcdir, 'models', 'index.rst')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("\n".join(tree_lines) + "\n")

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
    'matplotlib.sphinxext.plot_directive',
    'jupyter_sphinx',
    'IPython.sphinxext.ipython_directive',
]

plot_html_show_formats = False
plot_html_show_source_link = False

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

math_dollar_node_blacklist = NODE_BLACKLIST + (sphinx.addnodes.pending_xref_condition,)

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
    if getattr(obj, "_pmrf_auto", False):
        return True
    
    func = getattr(obj, "__func__", None)
    if func and getattr(func, "_pmrf_auto", False):
        return True

    if isinstance(obj, property):
        if getattr(obj.fget, "_pmrf_auto", False):
            return True

    return False


def skip_member(app, what, name, obj, skip, options):
    if is_pmrf_auto(obj):
        return True

    if isinstance(obj, types.ModuleType):
        return True

    obj_module = getattr(obj, '__module__', None)
    if obj_module is None and isinstance(obj, property):
        obj_module = getattr(obj.fget, '__module__', '')

    methods_to_hide = {"s", "y", "z", "a", "build"}
    if name in methods_to_hide:
        qualname = getattr(obj, '__qualname__', '')
        
        if not qualname:
            if isinstance(obj, property):
                qualname = getattr(obj.fget, '__qualname__', '')
            elif hasattr(obj, '__func__'):
                qualname = getattr(obj.__func__, '__qualname__', '')
        
        if qualname:
            parts = qualname.split('.')
            if len(parts) >= 2:
                parent_class_name = parts[-2]
                if parent_class_name != 'Model':
                    return True
        else:
            return True

    return skip

def setup(app):
    app.connect("builder-inited", generate_models_tree)
    app.connect("autodoc-skip-member", skip_member)