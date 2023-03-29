#!/usr/bin/env python
"""Script used to create the pickle files used in tests/test_pickle.py.

Note that the exact binary of the pickle files may differ depending on which
Python version is used to create them.
"""
import argparse
import pickle
from pathlib import Path

import graphblas as gb
from graphblas.tests.test_pickle import *


def pickle1(filepath):
    suitesparse = gb.backend == "suitesparse"
    v = gb.Vector.from_coo([1], 2)

    unary_pickle = gb.core.operator.UnaryOp.register_new("unary_pickle", unarypickle)
    binary_pickle = gb.core.operator.BinaryOp.register_new("binary_pickle", binarypickle)
    monoid_pickle = gb.core.operator.Monoid.register_new("monoid_pickle", binary_pickle, 0)
    semiring_pickle = gb.core.operator.Semiring.register_new(
        "semiring_pickle", monoid_pickle, binary_pickle
    )

    unary_anon = gb.core.operator.UnaryOp.register_anonymous(unaryanon)
    binary_anon = gb.core.operator.BinaryOp.register_anonymous(binaryanon)
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
        "unary_pickle": unary_pickle,
        "unary_pickle[UINT16]": unary_pickle["UINT16"],
        "binary_pickle": binary_pickle,
        "monoid_pickle": monoid_pickle,
        "semiring_pickle": semiring_pickle,
        "unary_anon": unary_anon,
        "unary_anon[float]": unary_anon[float],
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
    with filepath.open("wb") as f:
        pickle.dump(d, f)


def pickle2(filepath):
    unary_pickle = gb.core.operator.UnaryOp.register_new(
        "unary_pickle_par", unarypickle_par, parameterized=True
    )
    binary_pickle = gb.core.operator.BinaryOp.register_new(
        "binary_pickle_par", binarypickle_par, parameterized=True
    )
    monoid_pickle = gb.core.operator.Monoid.register_new("monoid_pickle_par", binary_pickle, 0)
    semiring_pickle = gb.core.operator.Semiring.register_new(
        "semiring_pickle_par", monoid_pickle, binary_pickle
    )

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
        "unary_pickle": unary_pickle,
        "binary_pickle": binary_pickle,
        "monoid_pickle": monoid_pickle,
        "semiring_pickle": semiring_pickle,
        "unary_pickle(0)": unary_pickle(0),
        "binary_pickle(0)": binary_pickle(0),
        "monoid_pickle(0)": monoid_pickle(0),
        "semiring_pickle(0)": semiring_pickle(0),
        "unary_pickle(0)[UINT16]": unary_pickle(0)["UINT16"],
    }
    with filepath.open("wb") as f:
        pickle.dump(d, f)


def pickle3(filepath):
    record_dtype = np.dtype([("a", np.bool_), ("b", np.int32)], align=True)
    udt = gb.dtypes.register_new("PickledUDT", record_dtype)

    np_dtype = np.dtype("(2,)uint32")
    udt2 = gb.dtypes.register_anonymous(np_dtype, "pickled_subdtype")

    v = gb.Vector(udt, size=2)
    v[0] = (False, 1)
    v[1] = (True, 3)

    A = gb.Matrix(udt2, nrows=1, ncols=2)
    A[0, 0] = (1, 2)
    A[0, 1] = (3, 4)
    d = {
        "PickledUDT": udt,
        "pickled_subdtype": udt2,
        "v": v,
        "A": A,
        "any[udt]": gb.binary.any[udt],
    }
    with filepath.open("wb") as f:
        pickle.dump(d, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices={"suitesparse", "suitesparse-vanilla"}, default="suitesparse"
    )
    args = parser.parse_args()
    gb.init(args.backend)
    if args.backend == "suitesparse-vanilla":
        extra = "-vanilla"
    else:
        extra = ""
    path = Path(gb.tests.__file__).parent
    pickle1(path / f"pickle1{extra}.pkl")
    pickle2(path / f"pickle2{extra}.pkl")
    pickle3(path / f"pickle3{extra}.pkl")
