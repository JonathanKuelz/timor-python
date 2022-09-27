# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from pathlib import Path
import os
import re
import sys
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------

project = 'Timor'
copyright = '2022, Jonathan Kuelz'
author = 'Jonathan Kuelz'

# Get the version
with Path('../../src/timor/__init__.py').open('r') as init_file:
	init_content = init_file.read()
version = re.match(r'__version__ = [\"\']([0-9.]*)[\"\']', init_content)[1]

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'autoapi.extension', 'sphinx.ext.intersphinx', 'sphinx_git']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['*tests']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['']

# Auto Doc
autodoc_typehints = 'description'  # Use typing information

# Link to external sphinx
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'typing': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy', None),
                       'matplotlib': ('https://matplotlib.org/stable', None)}


# Auto API doc
autoapi_type = 'python'
autoapi_dirs = [os.path.abspath('../../src')]
autoapi_ignore = []
autoapi_options = ['members',
                   'undoc-members',
                   'show-inheritance',
                   'show-module-summary',
                   'special-members',
                   'imported-members']
autoapi_keep_files = True
autoapi_python_class_content = 'both'  # Ensure that __init__'s parameters are completely shown
