# Python-graphblas

[![conda-forge](https://img.shields.io/conda/vn/conda-forge/python-graphblas.svg)](https://anaconda.org/conda-forge/python-graphblas)
[![pypi](https://img.shields.io/pypi/v/python-graphblas.svg)](https://pypi.python.org/pypi/python-graphblas/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/metagraph-dev/python-graphblas/blob/main/LICENSE)
[![Tests](https://github.com/metagraph-dev/python-graphblas/workflows/Tests/badge.svg?branch=main)](https://github.com/metagraph-dev/python-graphblas/actions)
[![Docs](https://readthedocs.org/projects/python-graphblas/badge/?version=latest)](https://python-graphblas.readthedocs.io/en/latest/)
[![Coverage](https://coveralls.io/repos/metagraph-dev/python-graphblas/badge.svg?branch=main)](https://coveralls.io/r/metagraph-dev/python-graphblas)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/metagraph-dev/python-graphblas/HEAD?filepath=notebooks%2FIntro%20to%20GraphBLAS%20%2B%20SSSP%20example.ipynb)

Python wrapper around GraphBLAS

To install, `conda install -c conda-forge python-graphblas` or `pip install python-graphblas`. This will also install the SuiteSparse `graphblas` compiled C library.

Currently works with [SuiteSparse:GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS), but the goal is to make it work with all implementations of the GraphBLAS spec.

The approach taken with this library is to follow the C-API specification as closely as possible while making improvements
allowed with the Python syntax. Because the spec always passes in the output object to be written to, we follow the same,
which is very different from the way Python normally operates. In fact, many who are familiar with other Python data
libraries (numpy, pandas, etc) will find it strange to not create new objects for every call.

At the highest level, the goal is to separate output, mask, and accumulator on the left side of the assignment
operator `=` and put the computation on the right side. Unfortunately, that approach doesn't always work very well
with how Python handles assignment, so instead we (ab)use the left-shift `<<` notation to give the same flavor of
assignment. This opens up all kinds of nice possibilities.

This is an example of how the mapping works:
```C
// C call
GrB_Matrix_mxm(M, mask, GrB_PLUS_INT64, GrB_MIN_PLUS_INT64, A, B, NULL)
```
```python
# Python call
M(mask.V, accum=binary.plus) << A.mxm(B, semiring.min_plus)
```

The expression on the right `A.mxm(B)` creates a delayed object which does no computation. Once it is used in the
`<<` expression with `M`, the whole thing is translated into the equivalent GraphBLAS call.

Delayed objects also have a `.new()` method which can be used to force computation and return a new
object. This is convenient and often appropriate, but will create many unnecessary objects if used in a loop. It
also loses the ability to perform accumulation with existing results. For best performance, following the standard
GraphBLAS approach of (1) creating the object outside the loop and (2) using the object repeatedly within each loop
is a much better approach, even if it doesn't feel very Pythonic.

Descriptor flags are set on the appropriate elements to keep logic close to what it affects. Here is the same call
with descriptor bits set. `ttcsr` indicates transpose the first and second matrices, complement the structure of the mask,
and do a replacement on the output.
```C
// C call
GrB_Matrix_mxm(M, mask, GrB_PLUS_INT64, GrB_MIN_PLUS_INT64, A, B, desc.ttcsr)
```
```python
# Python call
M(~mask.S, accum=binary.plus, replace=True) << A.T.mxm(B.T, semiring.min_plus)
```

The objects receiving the flag operations (A.T, ~mask, etc) are also delayed objects. They hold on to the state but
do no computation, allowing the correct descriptor bits to be set in a single GraphBLAS call.

**If no mask or accumulator is used, the call looks like this**:
```python
M << A.mxm(B, semiring.min_plus)
```
The use of `<<` to indicate updating is actually just syntactic sugar for a real `.update()` method. The above
expression could be written as:
```python
M.update(A.mxm(B, semiring.min_plus))
```

## Operations
```python
M(mask, accum) << A.mxm(B, semiring)        # mxm
w(mask, accum) << A.mxv(v, semiring)        # mxv
w(mask, accum) << v.vxm(B, semiring)        # vxm
M(mask, accum) << A.ewise_add(B, binaryop)  # eWiseAdd
M(mask, accum) << A.ewise_mult(B, binaryop) # eWiseMult
M(mask, accum) << A.kronecker(B, binaryop)  # kronecker
M(mask, accum) << A.T                       # transpose
```
## Extract
```python
M(mask, accum) << A[rows, cols]             # rows and cols are a list or a slice
w(mask, accum) << A[rows, col_index]        # extract column
w(mask, accum) << A[row_index, cols]        # extract row
s = A[row_index, col_index].value           # extract single element
```
## Assign
```python
M(mask, accum)[rows, cols] << A             # rows and cols are a list or a slice
M(mask, accum)[rows, col_index] << v        # assign column
M(mask, accum)[row_index, cols] << v        # assign row
M(mask, accum)[rows, cols] << s             # assign scalar to many elements
M[row_index, col_index] << s                # assign scalar to single element
                                            # (mask and accum not allowed)
del M[row_index, col_index]                 # remove single element
```
## Apply
```python
M(mask, accum) << A.apply(unaryop)
M(mask, accum) << A.apply(binaryop, left=s)   # bind-first
M(mask, accum) << A.apply(binaryop, right=s)  # bind-second
```
## Reduce
```python
v(mask, accum) << A.reduce_rowwise(op)      # reduce row-wise
v(mask, accum) << A.reduce_columnwise(op)   # reduce column-wise
s(accum) << A.reduce_scalar(op)
s(accum) << v.reduce(op)
```
## Creating new Vectors / Matrices
```python
A = Matrix.new(dtype, num_rows, num_cols)   # new_type
B = A.dup()                                 # dup
A = Matrix.from_values([row_indices], [col_indices], [values])  # build
```
## New from delayed
Delayed objects can be used to create a new object using `.new()` method
```python
C = A.mxm(B, semiring).new()
```
## Properties
```python
size = v.size                               # size
nrows = M.nrows                             # nrows
ncols = M.ncols                             # ncols
nvals = M.nvals                             # nvals
rindices, cindices, vals = M.to_values()    # extractTuples
```
## Initialization
There is a mechanism to initialize `graphblas` with a context prior to use. This allows for setting the backend to
use as well as the blocking/non-blocking mode. If the context is not initialized, a default initialization will
be performed automatically.
```python
import graphblas as gb
# Context initialization must happen before any other imports
gb.init('suitesparse', blocking=True)

# Now we can import other items from graphblas
from graphblas import binary, semiring
from graphblas import Matrix, Vector, Scalar
```
## Performant User Defined Functions
Python-graphblas requires `numba` which enables compiling user-defined Python functions to native C for use in GraphBLAS.

Example customized UnaryOp:
```python
from graphblas import unary
from graphblas.operator import UnaryOp

def force_odd_func(x):
    if x % 2 == 0:
        return x + 1
    return x

UnaryOp.register_new('force_odd', force_odd_func)

v = Vector.from_values([0, 1, 3], [1, 2, 3])
w = v.apply(unary.force_odd).new()
w  # indexes=[0, 1, 3], values=[1, 3, 3]
```
Similar methods exist for BinaryOp, Monoid, and Semiring.

## Import/Export connectors to the Python ecosystem
`graphblas.io` contains functions for converting to and from:
```python
import graphblas as gb

# numpy arrays
# 1-D array becomes Vector, 2-D array becomes Matrix
A = gb.io.from_numpy(m)
m = gb.io.to_numpy(A)

# scipy.sparse matrices
A = gb.io.from_scipy_sparse_matrix(m)
m = gb.io.to_scipy_sparse_matrix(m, format='csr')

# networkx graphs
A = gb.io.from_networkx(g)
g = gb.io.to_networkx(A)
```
