import multiprocessing
import pickle
from pathlib import Path

import numpy as np
import pytest

import graphblas as gb
from graphblas.core import _supports_udfs as supports_udfs  # noqa: F401

suitesparse = gb.backend == "suitesparse"


def unarypickle(x):
    return x + 1  # pragma: no cover (numba)


def binarypickle(x, y):
    return x + y  # pragma: no cover (numba)


def unaryanon(x):
    return x + 2  # pragma: no cover (numba)


def binaryanon(x, y):
    return x + y  # pragma: no cover (numba)


def indexunaryanon(x, row, col, thunk):
    return x >= thunk  # pragma: no cover (numba)


@pytest.fixture
def extra():
    if gb.backend == "suitesparse-vanilla":
        return "-vanilla"
    return ""


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_deserialize(extra):
    path = Path(__file__).parent / f"pickle1{extra}.pkl"
    with path.open("rb") as f:
        d = pickle.load(f)
    check_values(d)
    # Again!
    with path.open("rb") as f:
        d = pickle.load(f)
    check_values(d)

    # Now that these operators have been registered, let's take them for a run
    d2 = {
        "unary_pickle": gb.unary.unary_pickle,
        "binary_pickle": gb.binary.binary_pickle,
        "monoid_pickle": gb.monoid.monoid_pickle,
        "semiring_pickle": gb.semiring.semiring_pickle,
    }
    pkl = pickle.dumps(d2)
    d3 = pickle.loads(pkl)
    assert d3["unary_pickle"] is gb.unary.unary_pickle
    assert d3["binary_pickle"] is gb.binary.binary_pickle
    assert d3["monoid_pickle"] is gb.monoid.monoid_pickle
    assert d3["semiring_pickle"] is gb.semiring.semiring_pickle


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_serialize():
    v = gb.Vector.from_coo([1], 2)

    # unary_pickle = gb.core.operator.UnaryOp.register_new("unary_pickle", unarypickle)
    # binary_pickle = gb.core.operator.BinaryOp.register_new("binary_pickle", binarypickle)
    # monoid_pickle = gb.core.operator.Monoid.register_new("monoid_pickle", binary_pickle, 0)
    # semiring_pickle = gb.core.operator.Semiring.register_new(
    #     "semiring_pickle", monoid_pickle, binary_pickle
    # )

    unary_anon = gb.core.operator.UnaryOp.register_anonymous(unaryanon)
    binary_anon = gb.core.operator.BinaryOp.register_anonymous(binaryanon)
    indexunary_anon = gb.core.operator.IndexUnaryOp.register_anonymous(indexunaryanon)
    select_anon = gb.core.operator.SelectOp.register_anonymous(indexunaryanon)
    monoid_anon = gb.core.operator.Monoid.register_anonymous(binary_anon, 0)
    semiring_anon = gb.core.operator.Semiring.register_anonymous(monoid_anon, binary_anon)
    d = {
        "scalar": gb.Scalar.from_value(2),
        "empty_scalar": gb.Scalar(bool),
        "vector": v,
        "matrix": gb.Matrix.from_coo([2], [3], 4),
        "matrix.T": gb.Matrix.from_coo([3], [4], 5).T,
        "vector.S": v.S,
        "vector.V": v.V,
        "~vector.S": ~v.S,
        "~vector.V": ~v.V,
        "unary.abs": gb.unary.abs,
        "binary.minus": gb.binary.minus,
        "monoid.lxor": gb.monoid.lxor,
        "semiring.plus_times": gb.semiring.plus_times,
        "unary.abs[int]": gb.unary.abs[int],
        "binary.minus[float]": gb.binary.minus[float],
        "monoid.lxor[bool]": gb.monoid.lxor[bool],
        "semiring.plus_times[float]": gb.semiring.plus_times[float],
        "unary.numpy.spacing": gb.unary.numpy.spacing,
        "binary.numpy.gcd": gb.binary.numpy.gcd,
        "monoid.numpy.logaddexp": gb.monoid.numpy.logaddexp,
        "semiring.numpy.logaddexp2_hypot": gb.semiring.numpy.logaddexp2_hypot,
        "unary.numpy.spacing[float]": gb.unary.numpy.spacing[float],
        "binary.numpy.gcd[int]": gb.binary.numpy.gcd[int],
        "monoid.numpy.logaddexp[float]": gb.monoid.numpy.logaddexp[float],
        "semiring.numpy.logaddexp2_hypot[float]": gb.semiring.numpy.logaddexp2_hypot[float],
        "agg.sum": gb.agg.sum,
        "binary.absfirst": gb.binary.absfirst,
        "binary.absfirst[float]": gb.binary.absfirst[float],
        "binary.isclose": gb.binary.isclose,
        # "unary_pickle": unary_pickle,
        # "unary_pickle[UINT16]": unary_pickle["UINT16"],
        # "binary_pickle": binary_pickle,
        # "monoid_pickle": monoid_pickle,
        # "semiring_pickle": semiring_pickle,
        "unary_anon": unary_anon,
        "unary_anon[float]": unary_anon[float],
        "indexunary_anon": indexunary_anon,
        "select_anon": select_anon,
        "binary_anon": binary_anon,
        "monoid_anon": monoid_anon,
        "semiring_anon": semiring_anon,
        "dtypes.BOOL": gb.dtypes.BOOL,
        "dtypes._INDEX": gb.dtypes._INDEX,
        "all_indices": gb.core.expr._ALL_INDICES,
        "replace": gb.replace,
    }
    if suitesparse:
        d["agg.ss.first[int]"] = gb.agg.ss.first[int]
    try:
        pkl = pickle.dumps(d)
    except Exception:  # pragma: no cover (debug)
        for key, val in d.items():
            try:
                pickle.dumps(val)
            except Exception as exc:
                print(f"Pickle error: {key} {exc!r}")
        raise
    d2 = pickle.loads(pkl)
    check_values(d)
    check_values(d2)


def check_values(d):
    v = gb.Vector.from_coo([1], 2)
    assert d["scalar"].isequal(gb.Scalar.from_value(2), check_dtype=True)
    assert d["empty_scalar"].isequal(gb.Scalar(bool), check_dtype=True)
    assert d["vector"].isequal(v, check_dtype=True)
    assert d["matrix"].isequal(gb.Matrix.from_coo([2], [3], 4), check_dtype=True)
    assert d["matrix.T"].isequal(gb.Matrix.from_coo([3], [4], 5).T, check_dtype=True)
    assert type(d["vector.S"]) is gb.core.mask.StructuralMask
    assert d["vector.S"].parent.isequal(v, check_dtype=True)
    assert type(d["vector.V"]) is gb.core.mask.ValueMask
    assert d["vector.V"].parent.isequal(v, check_dtype=True)
    assert type(d["~vector.S"]) is gb.core.mask.ComplementedStructuralMask
    assert d["~vector.S"].parent.isequal(v, check_dtype=True)
    assert type(d["~vector.V"]) is gb.core.mask.ComplementedValueMask
    assert d["~vector.V"].parent.isequal(v, check_dtype=True)
    assert d["unary.abs"] is gb.unary.abs
    assert d["binary.minus"] is gb.binary.minus
    assert d["monoid.lxor"] is gb.monoid.lxor
    assert d["semiring.plus_times"] is gb.semiring.plus_times
    assert d["unary.abs[int]"] is gb.unary.abs[int]
    assert d["binary.minus[float]"] is gb.binary.minus[float]
    assert d["monoid.lxor[bool]"] is gb.monoid.lxor[bool]
    assert d["semiring.plus_times[float]"] is gb.semiring.plus_times[float]
    assert d["unary.numpy.spacing"] is gb.unary.numpy.spacing
    assert d["binary.numpy.gcd"] is gb.binary.numpy.gcd
    assert d["monoid.numpy.logaddexp"] is gb.monoid.numpy.logaddexp
    assert d["semiring.numpy.logaddexp2_hypot"] is gb.semiring.numpy.logaddexp2_hypot
    assert d["unary.numpy.spacing[float]"] is gb.unary.numpy.spacing[float]
    assert d["binary.numpy.gcd[int]"] is gb.binary.numpy.gcd[int]
    assert d["monoid.numpy.logaddexp[float]"] is gb.monoid.numpy.logaddexp[float]
    assert d["semiring.numpy.logaddexp2_hypot[float]"] is gb.semiring.numpy.logaddexp2_hypot[float]
    assert d["agg.sum"] is gb.agg.sum
    if suitesparse and "agg.ss.first[int]" in d:
        assert d["agg.ss.first[int]"] is gb.agg.ss.first[int]
    assert d["binary.absfirst"] is gb.binary.absfirst
    assert d["binary.absfirst[float]"] is gb.binary.absfirst[float]
    assert d["binary.isclose"] is gb.binary.isclose
    if "unary_pickle" in d:
        assert d["unary_pickle"] is gb.unary.unary_pickle
    if "unary_pickle[UINT16]" in d:
        assert d["unary_pickle[UINT16]"] is gb.unary.unary_pickle["UINT16"]
    if "binary_pickle" in d:
        assert d["binary_pickle"] is gb.binary.binary_pickle
    if "monoid_pickle" in d:
        assert d["monoid_pickle"] is gb.monoid.monoid_pickle
    if "semiring_pickle" in d:
        assert d["semiring_pickle"] is gb.semiring.semiring_pickle
    d["unary_anon"]
    d["unary_anon[float]"]
    d["binary_anon"]
    d["monoid_anon"]
    d["semiring_anon"]
    assert d["dtypes.BOOL"] is gb.dtypes.BOOL
    assert d["dtypes._INDEX"] is gb.dtypes._INDEX
    assert d["all_indices"] is gb.core.expr._ALL_INDICES
    assert d["replace"] is gb.replace


def unarypickle_par(x):
    def inner(y):
        return x + y  # pragma: no cover (numba)

    return inner


def binarypickle_par(z):
    def inner(x, y):
        return x + y + z  # pragma: no cover (numba)

    return inner


def unaryanon_par(x):
    def inner(y):
        return y + x  # pragma: no cover (numba)

    return inner


def binaryanon_par(z):
    def inner(x, y):
        return x + y + z  # pragma: no cover (numba)

    return inner


def identity_par(z):
    return -z


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_serialize_parameterized():
    # unary_pickle = gb.core.operator.UnaryOp.register_new(
    #     "unary_pickle_par", unarypickle_par, parameterized=True
    # )
    # binary_pickle = gb.core.operator.BinaryOp.register_new(
    #     "binary_pickle_par", binarypickle_par, parameterized=True
    # )
    # monoid_pickle = gb.core.operator.Monoid.register_new("monoid_pickle_par", binary_pickle, 0)
    # semiring_pickle = gb.core.operator.Semiring.register_new(
    #     "semiring_pickle_par", monoid_pickle, binary_pickle
    # )

    unary_anon = gb.core.operator.UnaryOp.register_anonymous(unaryanon_par, parameterized=True)
    binary_anon = gb.core.operator.BinaryOp.register_anonymous(binaryanon_par, parameterized=True)
    monoid_anon = gb.core.operator.Monoid.register_anonymous(binary_anon, 0)
    monoid2_anon = gb.core.operator.Monoid.register_anonymous(binary_anon, identity_par)
    semiring_anon = gb.core.operator.Semiring.register_anonymous(monoid_anon, binary_anon)
    d = {
        "binary.isclose(rel_tol=1., abs_tol=1.)": gb.binary.isclose(rel_tol=1.0, abs_tol=1.0),
        "unary_anon": unary_anon,
        "binary_anon": binary_anon,
        "monoid_anon": monoid_anon,
        "monoid2_anon": monoid2_anon,
        "semiring_anon": semiring_anon,
        "unary_anon(0)": unary_anon(0),
        "binary_anon(0)": binary_anon(0),
        "monoid_anon(0)": monoid_anon(0),
        "monoid2_anon(0)": monoid2_anon(0),
        "semiring_anon(0)": semiring_anon(0),
        "unary_anon(0)[int]": unary_anon(0)[int],
        # "unary_pickle": unary_pickle,
        # "binary_pickle": binary_pickle,
        # "monoid_pickle": monoid_pickle,
        # "semiring_pickle": semiring_pickle,
        # "unary_pickle(0)": unary_pickle(0),
        # "binary_pickle(0)": binary_pickle(0),
        # "monoid_pickle(0)": monoid_pickle(0),
        # "semiring_pickle(0)": semiring_pickle(0),
        # "unary_pickle(0)[UINT16]": unary_pickle(0)["UINT16"],
    }
    try:
        pkl = pickle.dumps(d)
    except Exception:  # pragma: no cover (debug)
        for key, val in d.items():
            try:
                pickle.dumps(val)
            except Exception as exc:
                print(f"Pickle error: {key} {exc!r}")
        raise
    pickle.loads(pkl)  # TODO: check results


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_deserialize_parameterized(extra):
    path = Path(__file__).parent / f"pickle2{extra}.pkl"
    with path.open("rb") as f:
        pickle.load(f)  # TODO: check results
    # Again!
    with path.open("rb") as f:
        pickle.load(f)  # TODO: check results


@pytest.mark.skipif("not supports_udfs")
def test_udt(extra):
    record_dtype = np.dtype([("x", np.bool_), ("y", np.int64)], align=True)
    udt = gb.dtypes.register_new("PickleUDT", record_dtype)
    assert not udt._is_anonymous
    assert pickle.loads(pickle.dumps(udt)) is udt

    np_dtype = np.dtype("(3,)uint16")
    udt2 = gb.dtypes.register_anonymous(np_dtype, "pickling")
    assert udt2._is_anonymous
    assert pickle.loads(pickle.dumps(udt2)).np_type == udt2.np_type

    path = Path(__file__).parent / f"pickle3{extra}.pkl"
    with path.open("rb") as f:
        d = pickle.load(f)
    udt3 = d["PickledUDT"]
    v = d["v"]
    assert udt3.name == "PickledUDT"
    assert udt3 is gb.dtypes.PickledUDT
    assert v.dtype == udt3
    expected = gb.Vector(udt3, size=2)
    expected[0] = (False, 1)
    expected[1] = (True, 3)
    assert expected.isequal(v)

    udt4 = d["pickled_subdtype"]
    A = d["A"]
    assert udt4.name == "pickled_subdtype"
    assert not hasattr(gb.dtypes, udt4.name)
    assert A.dtype == udt4
    expected = gb.Matrix(udt4, nrows=1, ncols=2)
    expected[0, 0] = (1, 2)
    expected[0, 1] = (3, 4)
    assert expected.isequal(A)

    any_udt = d["any[udt]"]
    assert any_udt is gb.binary.any[udt3]
    assert pickle.loads(pickle.dumps(gb.binary.first[udt, int])) is gb.binary.first[udt, int]


# Module-level so the spawn worker can pickle it; local functions can't cross processes.
def _crossproc_ibo_add_theta(x, ix, jx, y, iy, jy, theta):  # pragma: no cover (numba)
    return x + y + theta


# Operations dispatched by ``_crossproc_worker``. Each takes the unpickled dict
# and returns a JSON-friendly value to send through the Pipe.
def _op_udt_reduce(d):
    return d["vector"].reduce(d["op"]).new().value


def _op_bound_ibo_ewise(d):
    rows, cols, vals = d["A"].ewise_mult(d["B"], d["bound"]).new().to_coo()
    return list(rows), list(cols), [int(v) for v in vals]


_CROSSPROC_OPS = {
    "udt_reduce": _op_udt_reduce,
    "bound_ibo_ewise": _op_bound_ibo_ewise,
}


def _crossproc_worker(backend, op_name, payload, conn):
    """Run ``_CROSSPROC_OPS[op_name]`` on the unpickled payload in a fresh interpreter."""
    try:
        import graphblas as gb_

        gb_.init(backend)
        unpickled = pickle.loads(payload)
        conn.send(("ok", _CROSSPROC_OPS[op_name](unpickled)))
    except Exception as exc:  # pragma: no cover (only on failure)
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _run_crossproc(op_name, payload):
    """Spawn a child running ``_crossproc_worker`` and return ``rest`` from ``("ok", *rest)``."""
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_crossproc_worker,
        args=(gb.backend, op_name, payload, child_conn),
    )
    proc.start()
    child_conn.close()
    try:
        assert parent_conn.poll(timeout=120), "child process timed out"
        result = parent_conn.recv()
    finally:
        proc.join(timeout=5)
        if proc.is_alive():  # pragma: no cover (defensive)
            proc.terminate()
            proc.join(timeout=5)
    status, *rest = result
    assert status == "ok", f"child failed: {rest}"
    return rest


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_udt_pickle_crossprocess():
    """Verify a typed UDT monoid round-trips through a spawned child process.

    Same-process pickle already exercises the ``__reduce__`` and
    ``_deserialize`` paths. This test additionally proves the child can
    re-register the UDT and auto-lift the parent monoid for it without
    inheriting any state from the parent process; that's the case
    ``dask`` and ``multiprocessing`` actually hit.
    """
    record_dtype = np.dtype([("x", np.int64), ("y", np.int64)], align=True)
    udt_name = "CrossProcPickleUDT"
    # Use register_new so the child can re-register under the same name.
    if hasattr(gb.dtypes, udt_name):
        delattr(gb.dtypes, udt_name)
        gb.core.dtypes._registry.pop(udt_name, None)
    udt = gb.dtypes.register_new(udt_name, record_dtype)
    try:
        monoid = gb.monoid.plus[udt]
        v = gb.Vector(udt, size=3)
        v[0] = (1, 10)
        v[1] = (2, 20)
        v[2] = (3, 30)
        payload = pickle.dumps({"op": monoid, "vector": v})
        (value,) = _run_crossproc("udt_reduce", payload)
        # Reduce of (1,10), (2,20), (3,30) with plus = (6, 60).
        assert tuple(value) == (6, 60)
    finally:
        delattr(gb.dtypes, udt_name)
        gb.core.dtypes._registry.pop(udt_name, None)


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_bound_ibo_pickle_crossprocess():
    """A bound IBO must round-trip through a spawned child process.

    Bound IBOs pickle as ``(_rebind_indexbinaryop, (parent_ibo, type, theta))``.
    The parent IBO must re-register in the child (via the registered name) so
    the rebind can resolve it. Same-process pickle is already covered by
    ``test_pickle_bound`` in ``test_indexbinary.py``.
    """
    from graphblas.core.operator.indexbinary import _has_idxbinop

    if not _has_idxbinop:
        pytest.skip("requires SuiteSparse:GraphBLAS 9.4+")

    op_name = "CrossProcBoundIBO"
    if hasattr(gb.indexbinary, op_name):
        delattr(gb.indexbinary, op_name)
    gb.indexbinary.register_new(op_name, _crossproc_ibo_add_theta)
    try:
        bound = getattr(gb.indexbinary, op_name)[int](100)
        A = gb.Matrix.from_coo([0, 1], [0, 1], [3, 7])
        B = gb.Matrix.from_coo([0, 1], [0, 1], [5, 2])
        payload = pickle.dumps({"A": A, "B": B, "bound": bound})
        ((rows, cols, vals),) = _run_crossproc("bound_ibo_ewise", payload)
        # (0,0): 3 + 5 + 100 = 108 ; (1,1): 7 + 2 + 100 = 109
        assert rows == [0, 1]
        assert cols == [0, 1]
        assert vals == [108, 109]
    finally:
        delattr(gb.indexbinary, op_name)
