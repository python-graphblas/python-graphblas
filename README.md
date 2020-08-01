# grblas

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/grblas.svg)](https://anaconda.org/conda-forge/grblas)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/grblas.svg)](https://anaconda.org/conda-forge/grblas)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/metagraph-dev/grblas/blob/master/LICENSE)
[![Build Status](https://travis-ci.org/metagraph-dev/grblas.svg?branch=master)](https://travis-ci.org/metagraph-dev/grblas)
[![Coverage Status](https://coveralls.io/repos/metagraph-dev/grblas/badge.svg?branch=master)](https://coveralls.io/r/metagraph-dev/grblas)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python wrapper around GraphBLAS

To install, `conda install -c conda-forge grblas`. This will also install the SuiteSparse `graphblas` compiled C library.

Currently works with SuiteSparse:GraphBLAS, but the goal is to make it work with all implementations of the GraphBLAS spec.

The approach taken with this library is to follow the C-API specification as closely as possible while making improvements 
allowed with the Python syntax. Because the spec always passes in the output object to be written to, we follow the same, 
which is very different from the way Python normally operates. In fact, many who are familiar with other Python data 
libraries (numpy, pandas, etc) will find it strange to not create new objects for every call.

At the highest level, the goal is to separate output, mask, and accumulator on the left side of the assignment 
operator `=` and put the computation on the right side. Unfortunately, that approach doesn't always work very well
with how Python handles assignment, so instead we (ab)use the left-shift `<<` notation to give the same flavor of
assignment. This opens up all kinds of nice possibilities.

This is an example of how the mapping works:<br>
C call: `GrB_Matrix_mxm(M, mask, GrB_PLUS_INT64, GrB_MIN_PLUS_INT64, A, B, NULL)`<br>
Python call: `M(mask.V, accum=binary.plus) << A.mxm(B, semiring.min_plus)`<br>

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

C call: `GrB_Matrix_mxm(M, mask, GrB_PLUS_INT64, GrB_MIN_PLUS_INT64, A, B, desc.ttcsr)`<br>
Python call: `M(~mask.S, accum=binary.plus, replace=True) << A.T.mxm(B.T, semiring.min_plus)`

The objects receiving the flag operations (A.T, ~mask, etc) are also delayed objects. They hold on to the state but 
do no computation, allowing the correct descriptor bits to be set in a single GraphBLAS call.

**If no mask or accumulator is used, the call looks like this**:<br>
`M << A.mxm(B, semiring.min_plus)`

The use of `<<` to indicate updating is actually just syntactic sugar for a real `.update()` method. The above
expression could be written as:<br>
`M.update(A.mxm(B, semiring.min_plus))`

# Operations
 - mxm: `M(mask, accum) << A.mxm(B, semiring)`
 - mxv: `w(mask, accum) << A.mxv(v, semiring)`
 - vxm: `w(mask, accum) << v.vxm(B, semiring)`
 - eWiseAdd: `M(mask, accum) << A.ewise_add(B, binaryop)`
 - eWiseMult: `M(mask, accum) << A.ewise_mult(B, binaryop)`
 - extract: 
   + `M(mask, accum) << A[rows, cols]`  # rows and cols are a list or a slice
   + `w(mask, accum) << A[rows, col_index]`  # extract column
   + `w(mask, accum) << A[row_index, cols]`  # extract row
   + `s = A[row_index, col_index].value`  # extract single element
 - assign:
   + `M[rows, cols](mask, accum) << A`  # rows and cols are a list or a slice
   + `M[rows, col_index](mask, accum) << v`  # assign column
   + `M[row_index, cols](mask, accum) << v`  # assign row
   + `M[rows, cols](mask, accum) << s`  # assign scalar to many elements
   + `M[row_index, col_index] << s`  # assign scalar to single element (mask and accum not allowed)
   + `del M[row_index, col_index]`  # remove single element
 - apply:
   + `M(mask, accum) << A.apply(unaryop)`
   + `M(mask, accum) << A.apply(binaryop, left=s)`  # bind-first
   + `M(mask, accum) << A.apply(binaryop, right=s)`  # bind-second
 - reduce: 
   + `v(mask, accum) << A.reduce_rows(op)`  # reduce row-wise
   + `v(mask, accum) << A.reduce_columns(op)`  # reduce column-wise
   + `s(accum) << A.reduce_scalar(op)`
   + `s(accum) << v.reduce(op)`
 - transpose: `M(mask, accum) << A.T`
 - kronecker: `M(mask, accum) << A.kronecker(B, binaryop)`

# Creating new Vectors / Matrices
 - new_type: `A = Matrix.new(dtype, num_rows, num_cols)`
 - dup: `B = A.dup()`
 - build: `A = Matrix.from_values([row_indices], [col_indices], [values])`
 - new from delayed:
   - Delayed objects can be used to create a new object using `.new()` method
   - `C = A.mxm(B, semiring).new()`

# Properties
 - size: `size = v.size`
 - nrows: `nrows = M.nrows`
 - ncols: `ncols = M.ncols`
 - nvals: `nvals = M.nvals`
 - extractTuples: `rindices, cindices, vals = M.to_values()`

# Initialization
There is a mechanism to initialize `grblas` with a context prior to use. This allows for setting the backend to
use as well as the blocking/non-blocking mode. If the context is not initialized, a default initialization will
be performed automatically. 

```
import grblas
# Context initialization must happen before any other imports
grblas.init('suitesparse', blocking=True)

# Now we can import other items from grblas
from grblas import binary, semiring
from grblas import Matrix, Vector, Scalar
```

# Performant User Defined Functions
`grblas` requires `numba` which enables compiling user-defined Python functions to native C for use in GraphBLAS.

Example customized UnaryOp:
```
from grblas import unary
from grblas.ops import UnaryOp

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

# Import/Export connectors to the Python ecosystem
`grblas.io` contains functions for converting to and from:
- numpy arrays and matrices
  - `from_numpy(m)`  (_1-D array becomes Vector, 2-D array or matrix becomes Matrix_)
  - `to_numpy(g, format='array')`
- scipy.sparse matrices
  - `from_scipy_sparse_matrix(m)`
  - `to_scipy_sparse_matrix(m, format='csr')`
- networkx graphs
  - `from_networkx(g)`
  - `to_networkx(g)`

# Attribution
This library borrows some great ideas from [pygraphblas](https://github.com/michelp/pygraphblas),
especially around parsing operator names from SuiteSparse and the concept of a Scalar which the backend
implementation doesn't need to know about.

