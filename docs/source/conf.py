# docs/source/conf.py
import os
import sys

# Compute repo roots relative to this conf.py file
_here = os.path.abspath(os.path.dirname(__file__))          # docs/source
_repo_root = os.path.abspath(os.path.join(_here, '..', '..'))  # repo_root
_src_root = os.path.join(_repo_root, 'src')                 # repo_root/src

for p in (_src_root, _repo_root):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

project = 'paramrf'
author = 'Gary Allen'
release = '0.4.2'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',  # keep if you have any .md pages
]

autosummary_generate = True
autoclass_content = 'both'
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "property-doc-from-class": True,   # <--- important
}


napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True   # use :ivar: style instead of Attributes table

myst_enable_extensions = ['colon_fence', 'deflist', 'linkify']
myst_heading_anchors = 3

templates_path = ['_templates']
exclude_patterns = []
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

def skip_member(app, what, name, obj, skip, options):
    if what == "class" and getattr(obj, "_pmrf_auto", False):
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_member)