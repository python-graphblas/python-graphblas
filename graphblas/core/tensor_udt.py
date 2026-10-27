"""TTM by one SuiteSparse product with array-UDT values.

The factor's K rows become a full Vector of K values of a type of R doubles; one ``mxv`` of
the stored Matrix (the contracted axis as its columns) with ``plus_times[FP64, FP64[R]]``
gives a Vector holding, for each nonempty row, the R values of its output fiber. Those are
the output's rows, so the Vector's arrays become the (rows x R) FP64 Matrix through the
Containers without a copy of the values.

The type and ops are made here, not through ``graphblas.dtypes`` and ``graphblas.binary``:
- the type has a fixed C name per R (``_gbtensor_fp64x16``), where an anonymous UDT gets a
  per-process ``_gbudt_N`` (by registration order), so compiled kernels are found again in
  the JIT cache by every process, instead of being recompiled and overwritten. The name is
  claimed in ``udt_codegen._jit_c_type_defs``, as a registered UDT's is, so a user type of
  that name and another layout gets another C name (or, registered first, makes this one
  take another);
- the ops have a NULL function pointer and the C definitions ``udt_codegen`` generates for
  the lifted ops (``GxB_BinaryOp_new`` has the JIT compile the function), so no Numba cfunc
  is involved and the engine works without Numba.

It needs the JIT: without it the ops cannot be created (the product would run SuiteSparse's
generic kernel, about 2x slower than the other engines). So once an engine has chosen to run
(``jit_ready``), it raises the JIT control from SuiteSparse's default "run" to "on" as UDT
auto-lift does; an engine that declines a case leaves the control alone (``jit_possible``).
It is not used under an explicit "off". The type and ops of each R are made once per process.

A compile can still fail with the control "on" (a compiler that stopped working, a full
disk, an unwritable cache): SuiteSparse then returns ``GxB_JIT_ERROR`` rather than run the
generic kernel, and drops the control to "load" (``GB_jitifyer.c``). ``run`` catches that:
the control goes back to what it was before the engine raised it, a ``NoJITWarning`` says
what happened, the engine stays off for the rest of the process, and the caller takes
another engine.
"""

import warnings
from types import SimpleNamespace

import numpy as np

from ..dtypes import FP64
from . import NULL, ffi, lib
from . import tensor_layout as tl

# R -> (GrB_Type, GrB_Semiring), or None when they could not be made with the JIT on
_SEMIRINGS = {}
# After a failed compile: (the control before the engine raised it, the one SuiteSparse
# left, the error); the engine is then off for the rest of the process
_FAILURE = None


def jit_control():
    from .. import ss

    return ss.config["jit_c_control"]


def jit_possible():
    """Whether ``jit_ready`` may say yes, with none of its side effects (it may raise the
    control and probe the compiler): False after a failed compile, under "off", "pause" or
    "load", and when this process is known to have no working compiler. An engine checks
    this before deciding, and calls ``jit_ready`` only once it has decided to run.
    """
    from .ss import jit_config

    return (
        _FAILURE is None
        and jit_control() not in {"off", "pause", "load"}
        and jit_config._jit_enabled_for_udt is not False
    )


def jit_ready():
    """Whether a kernel armed now will be JIT-compiled. As for UDT auto-lift
    (``jit_config._enable_jit_for_udt``), SuiteSparse's default "run" is raised to "on" when
    a compiler works; an explicit "off", "pause" or "load" is honored. False after a failed
    compile (``run``).
    """
    from .ss.jit_config import _enable_jit_for_udt

    return _FAILURE is None and _enable_jit_for_udt()


def run(before, func, *args, **kwargs):
    """``func(*args, **kwargs)``: engine code that may JIT-compile, with ``before`` the JIT
    control from before ``jit_ready`` raised it. None when a compile fails (see the module
    notes): the control is set back to ``before``, and a ``NoJITWarning`` says so.
    """
    global _FAILURE
    from ..exceptions import JitError, NoJITWarning, NullPointer

    try:
        return func(*args, **kwargs)
    except (JitError, NullPointer) as exc:
        # (an op made from a NULL function and C source reports a failed compile as either)
        left = jit_control()
        if left != before:
            from .. import ss

            ss.config["jit_c_control"] = before
        _FAILURE = (before, left, exc)
        warnings.warn(
            f"The Tensor's array-UDT engine could not compile its kernels with the SuiteSparse "
            f"JIT ({type(exc).__name__}: {exc}); it is off for the rest of this process, and "
            f"another engine runs. jit_c_control is back to {before!r} (SuiteSparse had set "
            f"{left!r}).",
            NoJITWarning,
            stacklevel=2,
        )
        return None


def _binary_op(defn, ztype, xtype, ytype):
    c_name, c_defn = defn
    op = ffi.new("GrB_BinaryOp*")
    rc = lib.GxB_BinaryOp_new(
        op, NULL, ztype, xtype, ytype, ffi.new("char[]", c_name.encode()),
        ffi.new("char[]", c_defn.encode()),
    )  # fmt: skip
    tl._check(rc, "GxB_BinaryOp_new")
    return op[0]


def semiring(R):
    """``(GrB_Type, GrB_Semiring)``: R doubles, and ``plus_times[FP64, FP64[R]]`` made of
    JIT-compiled ops; None when the JIT is not on or they cannot be made (no compiler).
    """
    if R in _SEMIRINGS:
        return _SEMIRINGS[R]
    if not jit_ready():
        return None  # (not remembered: the JIT may be turned on later)
    from .operator.udt_codegen import (
        _jit_c_type_defs,
        _make_jit_c_definition,
        _make_jit_c_mixed_definition,
        _synthetic_udt_counter,
        _udt_c_typedef,
    )

    np_type = np.dtype((np.float64, (R,)))
    name, typedef = _udt_c_typedef(f"_gbtensor_fp64x{R}", np_type)
    # SuiteSparse finds JIT kernels by the type's and ops' C names alone, so claim this one in
    # the registry that registered UDTs use: a user type of that name and another layout then
    # gets a fresh C name (and so do its ops). If one already holds it, take a fresh name.
    while _jit_c_type_defs.setdefault(name, typedef) != typedef:
        fresh = f"_gbtensor_fp64x{R}_{next(_synthetic_udt_counter)}"
        name, typedef = _udt_c_typedef(fresh, np_type)
    # (what the generators read of a DataType)
    row = SimpleNamespace(name=name, jit_c_name=name, np_type=np_type)
    f64 = np.dtype(np.float64)
    times = _make_jit_c_mixed_definition("times", "*", FP64, row, row, [(f64, f64)])
    plus = _make_jit_c_definition("plus", "+", row, 2)
    rv = None
    try:
        t = ffi.new("GrB_Type*")
        tl._check(
            lib.GxB_Type_new(
                t, np_type.itemsize, ffi.new("char[]", name.encode()),
                ffi.new("char[]", typedef.encode()),
            ),
            "GxB_Type_new",
        )  # fmt: skip
        mult = _binary_op(times, t[0], FP64._carg, t[0])
        add = _binary_op(plus, t[0], t[0], t[0])
        monoid = ffi.new("GrB_Monoid*")
        tl._check(lib.GrB_Monoid_new_UDT(monoid, add, ffi.new("double[]", R)), "GrB_Monoid_new")
        sr = ffi.new("GrB_Semiring*")
        tl._check(lib.GrB_Semiring_new(sr, monoid[0], mult), "GrB_Semiring_new")
        rv = (t[0], sr[0])
    finally:
        # (a failed compile raises, for ``run``; this R is not tried again)
        _SEMIRINGS[R] = rv
    return rv


class _Vector:
    """A GrB_Vector of ``n`` values of ``gtype``, freed on exit."""

    def __init__(self, gtype, n):
        self.p = ffi.new("GrB_Vector*")
        tl._check(lib.GrB_Vector_new(self.p, gtype, n), "GrB_Vector_new")

    def __enter__(self):
        return self.p[0]

    def __exit__(self, *exc):
        lib.GrB_Vector_free(self.p)


class _UnloadedVector:
    """Context manager: GrB_Vector ``v`` unloaded into a GxB_Container, loaded back on exit
    (with ``load``) or left empty.
    """

    def __init__(self, v, *, load):
        self.v = v
        self.load = load

    def __enter__(self):
        tl._check(lib.GrB_Vector_wait(self.v, lib.GrB_MATERIALIZE), "GrB_Vector_wait")
        self._Cp = ffi.new("GxB_Container*")
        tl._check(lib.GxB_Container_new(self._Cp), "GxB_Container_new")
        rc = lib.GxB_unload_Vector_into_Container(self.v, self._Cp[0], NULL)
        if rc != 0:  # pragma: no cover (safety)
            lib.GxB_Container_free(self._Cp)
            tl._check(rc, "GxB_unload_Vector_into_Container")
        return self._Cp[0]

    def __exit__(self, *exc):
        rc = lib.GxB_load_Vector_from_Container(self.v, self._Cp[0], NULL) if self.load else 0
        lib.GxB_Container_free(self._Cp)
        tl._check(rc, "GxB_load_Vector_from_Container")


def _load(vec, arr, gtype, n=None):
    """Hand the numpy array ``arr`` (from numpy's allocator) to the Container vector ``vec``
    as ``n`` values of ``gtype`` (by default one per element).
    """
    from .utils import _give_buffer

    n = arr.size if n is None else n
    with _give_buffer(arr) as (ptr, nbytes, arena):
        tl._check(
            lib.GxB_Vector_load(vec, ptr, gtype, n, nbytes, lib.GrB_DEFAULT + arena, NULL),
            "GxB_Vector_load",
        )


def _retype(vec, gtype, n):
    """Make the Container vector ``vec`` hold its same bytes as ``n`` values of ``gtype``."""
    X = ffi.new("void**")
    t = ffi.new("GrB_Type*")
    count = ffi.new("uint64_t*")
    size = ffi.new("uint64_t*")
    handling = ffi.new("int*")
    tl._check(lib.GxB_Vector_unload(vec, X, t, count, size, handling, NULL), "GxB_Vector_unload")
    tl._check(lib.GxB_Vector_load(vec, X, gtype, n, size[0], handling[0], NULL), "GxB_Vector_load")


def ttm(M, U, desc=None):
    """``M @ U`` as an (nrows x R) FP64 Matrix held by row, by one ``mxv`` with values of R
    doubles: full rows where ``M`` has a nonempty row, none elsewhere (bitmap, or full, when
    at least half the rows are nonempty, else hypersparse). ``M`` is an FP64 Matrix held by row,
    ``U`` a C-contiguous (ncols x R) float64 array that is taken over. None when the engine
    is not available (no JIT), or when the product came back iso.
    """
    K, R = U.shape
    if (ops := semiring(R)) is None:
        return None
    row, sr = ops
    n = M._nrows
    nonempty = tl.count_nonempty_rows(M)
    with _Vector(row, K) as u, _Vector(row, n) as w:
        with _UnloadedVector(u, load=True) as cont:
            _load(cont.x, U.reshape(-1), row, K)
            cont.format = lib.GxB_FULL
            cont.nvals = K
            cont.iso = False
            cont.jumbled = False
        if 2 * nonempty < n:
            # A sparse output, so the product does not leave a bitmap of n values to convert
            # The empty Vector is sparse already: nothing converts now.
            # Otherwise the product's own bitmap or full result is kept: asking for bitmap here
            # would convert the empty Vector at once, allocating n values, and for a UDT that
            # conversion is a JIT kernel of its own (with a broken compiler, the call that
            # reported GxB_JIT_ERROR).
            tl._check(lib.GrB_Vector_set_INT32(w, lib.GxB_SPARSE, lib.GxB_SPARSITY_CONTROL), "set")
        tl._check(
            lib.GrB_mxv(w, NULL, NULL, sr, M.gb_obj[0], u, NULL if desc is None else desc._carg),
            "GrB_mxv",
        )
        return _rows_from_vector(w, n, R)


def _rows_from_vector(w, n, R):
    """The (n x R) FP64 Matrix whose row g holds the R doubles of ``w`` at g. ``w``'s values
    and indices move into it (O(1)); the hypersparse form adds a row pointer and the column
    indices, the bitmap one the bitmap of every value. ``w`` is left empty. None when ``w``
    is iso.
    """
    M = tl.new_matrix(FP64, n, R)
    with _UnloadedVector(w, load=False) as cw:
        fmt, nvals = cw.format, cw.nvals
        if cw.iso:
            # SuiteSparse marks a product of one entry by one value iso (a 1 x 1 Matrix with
            # one column in the factor): decline, and the products run instead
            return None
        with tl._Unloaded(M) as cm:
            _retype(cw.x, FP64._carg, (nvals if fmt == lib.GxB_SPARSE else n) * R)
            cm.x, cw.x = cw.x, cm.x
            if fmt == lib.GxB_SPARSE:
                cm.h, cw.i = cw.i, cm.h
                _load(cm.p, np.arange(0, (nvals + 1) * R, R, dtype=np.int64), lib.GrB_INT64)
                _load(cm.i, np.tile(np.arange(R, dtype=np.uint32), nvals), lib.GrB_UINT32)
                cm.format = lib.GxB_HYPERSPARSE
                cm.nrows_nonempty = nvals
            elif fmt == lib.GxB_BITMAP:
                with tl._VectorArray(cw.b, np.int8) as b:
                    bits = np.repeat(b.array, R)
                _load(cm.b, bits, lib.GrB_INT8)
                cm.format = lib.GxB_BITMAP
                cm.nrows_nonempty = -1
            else:
                cm.format = lib.GxB_FULL
                cm.nrows_nonempty = n
            cm.nvals = nvals * R
            cm.ncols_nonempty = -1
            cm.iso = False
            cm.jumbled = False
    return M
