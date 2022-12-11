.. _contributor_guide:

Contributor Guide
=================

.. _dev_workflow:

Development Workflow
--------------------

# 1. Creating a fork and a local copy:

   * Go to `https://github.com/python-graphblas/python-graphblas
     <https://github.com/python-graphblas/python-graphblas>`_ and click the
     "fork" button to create your own copy of the project.

   * Clone the project to your local computer::

      git clone git@github.com:your-username/python-graphblas.git

   * Navigate to the folder networkx and add the upstream repository::

      git remote add upstream git@github.com:python-graphblas/python-graphblas.git

   * Now, you have remote repositories named:

     - ``upstream``, which refers to the ``python-graphblas`` repository
     - ``origin``, which refers to your personal fork

# 2. Setting up a local development environment:

   * Next, you need to set up your build environment.
     Here are instructions for two popular environment managers:

    * ``venv`` (pip based)

       ::

         # Create a virtualenv named ``graphblas-dev`` that lives in the directory of
         # the same name
         python -m venv graphblas-dev
         # Activate it
         source graphblas-dev/bin/activate
         # Install main development and runtime dependencies of python-graphblas
         pip install -r requirements/default.txt -r requirements/test.txt -r requirements/developer.txt
         #
         # (Optional) Install pygraphviz and pydot packages
         # These packages require that you have your system properly configured
         # and what that involves differs on various systems.
         # pip install -r requirements/extra.txt
         #
         # Build and install networkx from source
         pip install -e .
         # Test your installation
         PYTHONPATH=. pytest networkx

     * ``conda`` (Anaconda or Miniconda)

       ::

         # Create a conda environment named ``graphblas-dev`` using environment.yml in the repository root
         conda create -f environment.yml
         # Activate it
         conda activate graphblas-dev
         # Install python-graphblas from source
         pip install -e .
         # Test your installation
         PYTHONPATH=. pytest graphblas

   * Finally, we recommend you use a pre-commit hook, which runs a number of tests when
     you type ``git commit``::

       pre-commit install
       # to trigger manual check use:
       # pre-commit run --all-files
