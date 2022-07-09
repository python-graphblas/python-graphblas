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
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "python-graphblas"
copyright = "2022, Anaconda, Inc"
author = "Anaconda, Inc"

# The full version, including alpha/beta/rc tags
# release = "1.3.2"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "nbsphinx", "sphinx_panels"]
html_css_files = ["custom.css"]
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

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/python-graphblas/python-graphblas",
}

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
