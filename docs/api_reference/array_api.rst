.. _api_array_api:

Array API Namespace (experimental)
----------------------------------

**THIS API IS EXPERIMENTAL AND MAY CHANGE.** See :ref:`array_api`.

``graphblas.array_api`` implements the `Python array API standard
<https://data-apis.org/array-api/2025.12/>`_ (2025.12). It has every function of the
standard but the ``fft`` extension, among them the four that 2025.12 added:
``broadcast_shapes``, ``isin``, ``linalg.eig`` and ``linalg.eigvals``. The functions of the
standard behave as the standard describes, so its pages are their reference.

This page lists the extensions: the Array's own attributes, the functions and keywords the
standard does not have, and the contractions, whose rules for fills are this namespace's
own. :ref:`tn_array_api_contractions` gives those rules exactly.

Array
~~~~~

.. autoclass:: graphblas.array_api.Array()
    :members: fill_value, nnz, tensor, todense, to_coo, to_scipy_sparse

Creating and changing fills
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: graphblas.array_api.asarray

.. autofunction:: graphblas.array_api.from_coo

.. autofunction:: graphblas.array_api.refill

.. autofunction:: graphblas.array_api.prune

Reductions that skip NaN
~~~~~~~~~~~~~~~~~~~~~~~~

numpy's ``nan`` reductions, with numpy's keywords. A library that looks for them on the
array's namespace, as xarray does for its float reductions (which skip NaN by default), uses
these instead of converting the Array to numpy.

.. autofunction:: graphblas.array_api.nansum

.. autofunction:: graphblas.array_api.nanprod

.. autofunction:: graphblas.array_api.nanmax

.. autofunction:: graphblas.array_api.nanmin

.. autofunction:: graphblas.array_api.nanmean

.. autofunction:: graphblas.array_api.nanvar

.. autofunction:: graphblas.array_api.nanstd

Contractions
~~~~~~~~~~~~

.. autofunction:: graphblas.array_api.matmul

.. autofunction:: graphblas.array_api.tensordot

.. autofunction:: graphblas.array_api.vecdot

.. autofunction:: graphblas.array_api.einsum

Deferred results
~~~~~~~~~~~~~~~~

.. autofunction:: graphblas.array_api.compute

Errors
~~~~~~

.. autoexception:: graphblas.array_api.DensifyError
