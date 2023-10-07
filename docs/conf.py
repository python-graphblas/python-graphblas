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
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))


# -- Project information -----------------------------------------------------

project = "python-graphblas"
copyright = "2020-2023, Anaconda, Inc. and contributors"
author = "Anaconda, Inc"

# The full version, including alpha/beta/rc tags
# release = "1.3.2"
# See: https://github.com/pypa/setuptools_scm/#usage-from-sphinx
from importlib.metadata import version  # noqa: E402 isort:skip

release = version("python-graphblas")
del version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "numpydoc", "sphinx_panels", "nbsphinx"]
html_css_files = ["custom.css", "matrix.css"]
html_js_files = ["custom.js"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_favicon = "_static/img/python-graphblas-logo.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_theme_options = {
    "logo": {
        "image_light": "_static/img/logo-horizontal-light.svg",
        "image_dark": "_static/img/logo-horizontal-dark.svg",
    },
    "github_url": "https://github.com/python-graphblas/python-graphblas",
}
html_show_sourcelink = False

autodoc_member_order = "groupwise"

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False


# -- Options for notebook output -------------------------------------------------

# nbsphinx config
nbsphinx_input_prompt = "%.0s"  # suppress prompt
nbsphinx_output_prompt = "%.0s"  # suppress prompt
nbsphinx_prolog = r"""
{% set nbname = env.doc2path(env.docname, base=False) %}

.. raw:: html


      <p class="text-right font-italic">
        This page was generated from
        <a href="../{{ nbname|e }}">{{ nbname|e }}</a>.
      </p>


.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ nbname | escape_latex }}}} \dotfill}}
"""
