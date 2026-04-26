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

* Clone the project to your local computer:

::

  git clone git@github.com:your-username/python-graphblas.git

* Navigate to the folder with python-graphblas and add the upstream repository:

::

  git remote add upstream git@github.com:python-graphblas/python-graphblas.git

* Now, you have remote repositories named:

- ``upstream``, which refers to the ``python-graphblas`` repository
- ``origin``, which refers to your personal fork

# 2. Setting up a local development environment:

* Next, you need to set up your build environment.

Here are instructions for two popular environment managers:

* ``venv`` (pip-based, only Linux supported currently)

::

  # Create a virtualenv named ``graphblas-dev`` that lives in the directory of
  # the same name
  python -m venv graphblas-dev
  # Activate it
  source graphblas-dev/bin/activate
  # Install main development and runtime dependencies of python-graphblas
  pip install -r dev-requirements.txt
  # Build and install python-graphblas from source
  pip install -e . --no-deps
  # Test your installation
  pytest graphblas

* ``conda`` (Anaconda or Miniconda)

::

  # Create a conda environment named ``graphblas-dev`` using environment.yml in the repository root
  conda env create -f environment.yml
  # Activate it
  conda activate graphblas-dev
  # Install python-graphblas from source
  pip install -e . --no-deps
  # Test your installation
  pytest graphblas

* Finally, we recommend you use a pre-commit hook, which runs a number of tests when you type ``git commit``:

::

  pre-commit install
  # to trigger manual check use:
  # pre-commit run --all-files

* If you are using `pixi <https://pixi.sh>`_, then you can use the following pre-defined tasks:

::

  # run pytest
  pixi run test

  # run pre-commit on all files
  pixi run pre-commit
