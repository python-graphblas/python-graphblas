# Tensor: backport candidates for Matrix and Vector

The Tensor is a new type, so it may go beyond Matrix and Vector behavior (`decisions.md`).
Where it does something Matrix and Vector could also do, this page lists it, so that the idea
is not lost and the two do not drift apart silently.

Each entry says what the Tensor does and what Matrix and Vector do today. "Today" was checked
by running python-graphblas on this branch, unless the entry says otherwise. Status:

- **on this branch**: the core code already exists here, in its own commit;
- **candidate**: not started.

## Candidates

1. **JIT C definitions for lifted ops between a scalar and an array or record UDT.**
   - Tensor: the array-UDT TTM engine uses them; `plus_times[FP64, FP64[R]]` compiles to one
     JIT kernel.
   - Without them, such ops run SuiteSparse's generic path with a Numba cfunc per element.
   - Status: **on this branch**, in core (`operator/udt_codegen.py`, `operator/binary.py`),
     as its own change. Matrix and Vector get it too, and it can go to main separately.
2. **Conversions as methods:** `from_pydata_sparse`, `to_pydata_sparse`, `from_scipy_sparse`,
   `to_scipy_sparse`.
   - Tensor: methods.
   - Matrix and Vector: only `graphblas.io` functions.
   - Status: candidate.
3. **A nonzero pydata `fill_value` raises.**
   - Tensor: raises ValueError, because a missing entry has no value.
   - Matrix and Vector: `graphblas.io.from_pydata_sparse` drops it silently (COO, 1-d and
     2-d), so the implicit values become missing.
   - Status: candidate, low priority. Whether to warn or raise, and whether to offer an
     opt-in, is undecided.
4. **`reduce(op, axis=..., keepdims=...)`** over any subset of axes.
   - Matrix: `reduce_rowwise`, `reduce_columnwise` and `reduce_scalar`.
   - Status: candidate, an `axis=` spelling on Matrix.
5. **`einsum` with any semiring.**
   - Tensor: `gb.einsum`. It accepts Matrix and Vector operands, and the result is a Tensor.
   - Matrix and Vector: none.
   - Status: candidate, a form that returns a Matrix or Vector.
6. **`take` (modes raise, wrap and clip), `take_along_axis` and `vindex`.**
   - Matrix and Vector: lists and ndarrays index as outer indices only.
   - Status: candidate.
7. **An opt-in raise for negative indices** (`wraparound=False`).
   - Matrix and Vector: negatives wrap once, with no option.
   - Status: candidate, small.
8. **Infix operators return `NotImplemented` for operand types they do not know.**
   - Tensor classes register in `utils._tensor_types`, and Matrix, Vector and Scalar defer to
     them.
   - For any other type, Matrix and Vector raise TypeError, so a third-party type's reflected
     operator never runs (checked: `A + Foo()` raises even when `Foo` defines `__radd__`).
   - Status: candidate. The design question is which types to defer to.
9. **Zero-copy Container load and unload, and an O(1) orientation flip.**
   - Tensor: used throughout `tensor_layout.py` (`_Unloaded`, `_VectorArray`,
     `flip_orientation`, `matrix_from_arrays`).
   - Matrix: `ss.unpack` widens 32-bit indices to 64-bit, and `ss.export` copies.
   - Status: candidate.
10. **Generated Numba kernels with a verified disk cache.**
    - Tensor: `tensor_numba._compile` writes a file per source hash and runs it only if its
      bytes match the generated source.
    - Matrix and Vector: Numba UDFs compile in every process.
    - Status: candidate, if other generated code appears.
11. **Not retrying after a failed compile.**
    - Tensor: the array-UDT engine leaves SuiteSparse's `"load"`, warns once, and turns
      itself off. (It used to restore the setting; with `"run"` back, auto-lift raised it to
      `"on"` again and the next UDT operation retried the compile and raised.)
    - UDT auto-lift leaves `"load"` in place after a failure too.
    - Status: now the same policy in both; a warning from auto-lift as well is a
      candidate.
12. **Lifted UDT operators without Numba when the JIT is on.**
    - Tensor: the engine makes its operators with `GxB_BinaryOp_new`, a NULL function pointer
      and the generated C definition.
    - Matrix and Vector: lifted operators are Numba cfuncs, with a C definition attached for
      the JIT.
    - Status: candidate. UDT operators would then work without Numba.
13. **Deterministic JIT C names for anonymous UDTs.**
    - Tensor: fixed names (`_gbtensor_fp64x16`).
    - Matrix and Vector: `_gbudt_N`, in registration order, so processes overwrite each
      other's cached JIT kernels.
    - Status: candidate, a name from a hash of the dtype.
14. **numpy-style `reshape`, `ravel` and `flatten`.**
    - Tensor: `reshape(..., copy=None)`, a view where the layout allows.
    - Matrix and Vector: no such methods (checked), only `M.ss.reshape`.
    - Status: candidate.
15. **Doctests of the user-guide page.**
    - Tensor: `test_user_guide_examples` runs every example of `docs/user_guide/tensor.rst`.
    - The other user-guide pages are not doctested.
    - Status: candidate (testing).
16. **`isequal` and `isclose` across kinds.**
    - Tensor: `T.isequal(A)` accepts a Matrix, Vector or Scalar of the same shape.
    - Matrix and Vector: `A.isequal(T)` defers to the Tensor, but a Matrix and a Vector (or
      a Scalar) still cannot be compared with each other: `A.isequal(v)` raises TypeError.
    - Status: candidate, for comparisons between Matrix, Vector and Scalar.

## Already the same, nothing to backport

- Broadcasting a Vector across a Matrix in element-wise ops: `A * v` multiplies column `j` by
  `v[j]`, numpy-style, as the Tensor does.
- A GraphBLAS Scalar as an index (`A[s, :]`, `v[s]`).
- Deleting a region (`del A[0:2, :]`).

Masks that combine a Tensor with a Matrix or Vector are Tensor-only: they make a Tensor mask.
