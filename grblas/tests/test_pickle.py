import os
import pickle

import pytest

import grblas as gb


def unarypickle(x):
    return x + 1


def binarypickle(x, y):
    return x + y


def unaryanon(x):
    return x + 2


def binaryanon(x, y):
    return x + y


@pytest.mark.slow
def test_deserialize():
    thisdir = os.path.dirname(__file__)
    with open(os.path.join(thisdir, "pickle1.pkl"), "rb") as f:
        d = pickle.load(f)
    check_values(d)


@pytest.mark.slow
def test_serialize():
    v = gb.Vector.from_values([1], 2)

    # unary_pickle = gb.operator.UnaryOp.register_new('unary_pickle', unarypickle)
    # binary_pickle = gb.operator.BinaryOp.register_new('binary_pickle', binarypickle)
    # monoid_pickle = gb.operator.Monoid.register_new('monoid_pickle', binary_pickle, 0)
    # semiring_pickle = gb.operator.Semiring.register_new('semiring_pickle', monoid_pickle, binary_pickle)

    unary_anon = gb.operator.UnaryOp.register_anonymous(unaryanon)
    binary_anon = gb.operator.BinaryOp.register_anonymous(binaryanon)
    monoid_anon = gb.operator.Monoid.register_anonymous(binary_anon, 0)
    semiring_anon = gb.operator.Semiring.register_anonymous(monoid_anon, binary_anon)

    d = {
        "scalar": gb.Scalar.from_value(2),
        "empty_scalar": gb.Scalar.new(bool),
        "vector": v,
        "matrix": gb.Matrix.from_values([2], [3], 4),
        "matrix.T": gb.Matrix.from_values([3], [4], 5).T,
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
        "agg.first[int]": gb.agg.first[int],
        "binary.absfirst": gb.binary.absfirst,
        "binary.absfirst[float]": gb.binary.absfirst[float],
        "binary.isclose": gb.binary.isclose,
        # 'binary.isclose(rel_tol=1., abs_tol=1.)': gb.binary.isclose(rel_tol=1., abs_tol=1.),
        # 'unary_pickle': unary_pickle,
        # 'unary_pickle[UINT16]': unary_pickle['UINT16'],
        # 'binary_pickle': binary_pickle,
        # 'monoid_pickle': monoid_pickle,
        # 'semiring_pickle': semiring_pickle,
        "unary_anon": unary_anon,
        "unary_anon[float]": unary_anon[float],
        "binary_anon": binary_anon,
        "monoid_anon": monoid_anon,
        "semiring_anon": semiring_anon,
        "dtypes.BOOL": gb.dtypes.BOOL,
        "dtypes._INDEX": gb.dtypes._INDEX,
        "all_indices": gb.expr._ALL_INDICES,
        "replace": gb.replace,
    }
    pkl = pickle.dumps(d)
    d2 = pickle.loads(pkl)

    check_values(d)
    check_values(d2)


def check_values(d):
    v = gb.Vector.from_values([1], 2)
    assert d["scalar"].isequal(gb.Scalar.from_value(2), check_dtype=True)
    assert d["empty_scalar"].isequal(gb.Scalar.new(bool), check_dtype=True)
    assert d["vector"].isequal(v, check_dtype=True)

    assert d["matrix"].isequal(gb.Matrix.from_values([2], [3], 4), check_dtype=True)
    assert d["matrix.T"].isequal(gb.Matrix.from_values([3], [4], 5).T, check_dtype=True)

    assert type(d["vector.S"]) is type(v.S)
    assert d["vector.S"].mask.isequal(v, check_dtype=True)
    assert type(d["vector.V"]) is type(v.V)
    assert d["vector.V"].mask.isequal(v, check_dtype=True)
    assert type(d["~vector.S"]) is type(~v.S)
    assert d["~vector.S"].mask.isequal(v, check_dtype=True)
    assert type(d["~vector.V"]) is type(~v.V)
    assert d["~vector.V"].mask.isequal(v, check_dtype=True)

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
    assert d["agg.first[int]"] is gb.agg.first[int]

    assert d["binary.absfirst"] is gb.binary.absfirst
    assert d["binary.absfirst[float]"] is gb.binary.absfirst[float]

    assert d["binary.isclose"] is gb.binary.isclose
    # 'binary.isclose(rel_tol=1., abs_tol=1.)'] is gb.binary.isclose(rel_tol=1., abs_tol=1.)

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
    assert d["all_indices"] is gb.expr._ALL_INDICES
    assert d["replace"] is gb.replace
