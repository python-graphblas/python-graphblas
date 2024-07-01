# These tests are very slow, since they force creation of all
# numpy unary, binary, monoid, and semiring objects.
import itertools
import sys

import numpy as np
import pytest

import graphblas as gb
import graphblas.binary.numpy as npbinary
import graphblas.monoid.numpy as npmonoid
import graphblas.semiring.numpy as npsemiring
import graphblas.unary.numpy as npunary
from graphblas import Vector, backend, config
from graphblas.core import _supports_udfs as supports_udfs
from graphblas.dtypes import _supports_complex

from .conftest import compute, shouldhave

is_win = sys.platform.startswith("win")
suitesparse = backend == "suitesparse"


def test_numpyops_dir():
    udf_or_mapped = supports_udfs or config["mapnumpy"]
    assert ("exp2" in dir(npunary)) == udf_or_mapped
    assert ("logical_and" in dir(npbinary)) == udf_or_mapped
    assert ("logaddexp" in dir(npmonoid)) == supports_udfs
    assert ("add_add" in dir(npsemiring)) == udf_or_mapped


@pytest.mark.skipif("not supports_udfs")
@pytest.mark.slow
def test_bool_doesnt_get_too_large():
    a = Vector.from_coo([0, 1, 2, 3], [True, False, True, False])
    b = Vector.from_coo([0, 1, 2, 3], [True, True, False, False])
    if gb.config["mapnumpy"]:
        with pytest.raises(KeyError, match="plus does not work with BOOL"):
            z = a.ewise_mult(b, gb.monoid.numpy.add).new()
    else:
        z = a.ewise_mult(b, gb.monoid.numpy.add).new()
        x, y = z.to_coo()
        np.testing.assert_array_equal(y, (True, True, True, False))

    def func(x):  # pragma: no cover (numba)
        return np.add(x, x)

    op = gb.core.operator.UnaryOp.register_anonymous(func)
    z = a.apply(op).new()
    x, y = z.to_coo()
    np.testing.assert_array_equal(y, (True, False, True, False))


@pytest.mark.slow
def test_npunary():
    L = list(range(5))
    data = [
        [Vector.from_coo([0, 1], [True, False]), np.array([True, False])],
        [Vector.from_coo(L, L), np.array(L, dtype=np.int64)],
        [Vector.from_coo(L, L, dtype="float64"), np.array(L, dtype=np.float64)],
    ]
    if _supports_complex:
        data.append(
            [Vector.from_coo(L, L, dtype="FC64"), np.array(L, dtype=np.complex128)],
        )
    blocklist = {
        "BOOL": {"negative", "positive", "reciprocal", "sign"},
        "INT64": {"reciprocal"},
        "FC64": {"ceil", "floor", "trunc"},
    }
    if suitesparse and is_win and gb.config["mapnumpy"]:
        # asin and asinh are known to be wrong in SuiteSparse:GraphBLAS
        # due to limitation of MSVC with complex
        blocklist["FC64"].update({"arcsin", "arcsinh"})
        blocklist["FC32"] = {"arcsin", "arcsinh"}
    if shouldhave(gb.binary, "isclose"):
        isclose = gb.binary.isclose(1e-6, 0)
    else:
        isclose = None
    for gb_input, np_input in data:
        for unary_name in sorted(npunary._unary_names & npunary.__dir__()):
            op = getattr(npunary, unary_name)
            if gb_input.dtype not in op.types or unary_name in blocklist.get(
                gb_input.dtype.name, ()
            ):
                continue
            if gb_input.dtype.name.startswith("FC"):
                # There are some nasty branch cuts as 1
                gb_input = gb_input.dup()
                gb_input[1] = 1.1 + 1.2j
                np_input = np_input.copy()
                np_input[1] = 1.1 + 1.2j
            with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
                gb_result = gb_input.apply(op).new()
                if gb_input.dtype == "BOOL" and gb_result.dtype == "FP32":
                    np_result = getattr(np, unary_name)(np_input, dtype="float32")
                    compare_op = isclose
                else:
                    np_result = getattr(np, unary_name)(np_input)
                    if gb_result.dtype.name.startswith("F"):
                        compare_op = isclose
                    else:
                        compare_op = npbinary.equal
            np_result = Vector.from_coo(
                list(range(np_input.size)), list(np_result), dtype=gb_result.dtype
            )
            assert gb_result.nvals == np_result.size
            if compare_op is None:
                continue  # FLAKY COVERAGE
            match = gb_result.ewise_mult(np_result, compare_op).new()
            if gb_result.dtype.name.startswith("F"):
                match(accum=gb.binary.lor) << gb_result.apply(npunary.isnan)
            compare = match.reduce(gb.monoid.land).new()
            if not compare:  # pragma: no cover (debug)
                if np.__version__.startswith("2.") and unary_name in {"sign"}:
                    # numba 0.60.0 does not match numpy 2.0
                    continue
                print(unary_name, gb_input.dtype)
                print(compute(gb_result))
                print(np_result)
            assert compare


@pytest.mark.slow
def test_npbinary():
    values1 = [0, 0, 1, 1, 2, 5]
    values2 = [0, 1, 0, 1, 3, 8]
    index = list(range(len(values1)))
    data = [
        [
            [Vector.from_coo(index, values1), Vector.from_coo(index, values2)],
            [np.array(values1, dtype=np.int64), np.array(values2, dtype=np.int64)],
        ],
        [
            [
                Vector.from_coo(index, values1, dtype="float64"),
                Vector.from_coo(index, values2, dtype="float64"),
            ],
            [np.array(values1, dtype=np.float64), np.array(values2, dtype=np.float64)],
        ],
        [
            [
                Vector.from_coo([0, 1, 2, 3], [True, False, True, False]),
                Vector.from_coo([0, 1, 2, 3], [True, True, False, False]),
            ],
            [np.array([True, False, True, False]), np.array([True, True, False, False])],
        ],
    ]
    if _supports_complex:
        data.append(
            [
                [
                    Vector.from_coo(index, values1, dtype="FC64"),
                    Vector.from_coo(index, values2, dtype="FC64"),
                ],
                [np.array(values1, dtype=np.complex128), np.array(values2, dtype=np.complex128)],
            ],
        )
    blocklist = {
        "FP64": {"floor_divide"},  # numba/numpy difference for 1.0 / 0.0
        "BOOL": {"gcd", "lcm", "subtract"},  # not supported by numpy
    }
    if shouldhave(gb.binary, "isclose"):
        isclose = gb.binary.isclose(1e-7, 0)
    else:
        isclose = None
    if shouldhave(npbinary, "equal"):
        equal = npbinary.equal
    else:
        equal = gb.binary.eq
    if shouldhave(npbinary, "isnan"):
        isnan = npunary.isnan
    else:
        isnan = gb.unary.isnan
    if shouldhave(npbinary, "isinf"):
        isinf = npunary.isinf
    else:
        isinf = gb.unary.isinf
    for (gb_left, gb_right), (np_left, np_right) in data:
        for binary_name in sorted(npbinary._binary_names & npbinary.__dir__()):
            op = getattr(npbinary, binary_name)
            if gb_left.dtype not in op.types or binary_name in blocklist.get(
                gb_left.dtype.name, ()
            ):
                continue
            if is_win and binary_name == "ldexp":
                # On Windows, the second argument must be int32 or less (I'm not sure why)
                np_right = np_right.astype(np.int32)
            with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
                gb_result = gb_left.ewise_mult(gb_right, op).new()
                try:
                    if gb_left.dtype == "BOOL" and gb_result.dtype == "FP32":
                        np_result = getattr(np, binary_name)(np_left, np_right, dtype="float32")
                        compare_op = isclose
                    else:
                        np_result = getattr(np, binary_name)(np_left, np_right)
                        if binary_name in {"arctan2"}:
                            compare_op = isclose
                        else:
                            compare_op = equal
                except Exception:  # pragma: no cover (debug)
                    print(f"Error computing numpy result for {binary_name}")
                    print(f"dtypes: ({gb_left.dtype}, {gb_right.dtype}) -> {gb_result.dtype}")
                    raise
            np_result = Vector.from_coo(np.arange(np_left.size), np_result, dtype=gb_result.dtype)

            assert gb_result.nvals == np_result.size
            if compare_op is None:
                continue  # FLAKY COVERAGE
            match = gb_result.ewise_mult(np_result, compare_op).new()
            if gb_result.dtype.name.startswith("F"):
                match(accum=gb.binary.lor) << gb_result.apply(isnan)
            if gb_result.dtype.name.startswith("FC"):
                # Divide by 0j sometimes result in different behavior, such as `nan` or `(inf+0j)`
                match(accum=gb.binary.lor) << gb_result.apply(isinf)
            compare = match.reduce(gb.monoid.land).new()
            if not compare:  # pragma: no cover (debug)
                print(compare_op)
                print(binary_name)
                print(compute(gb_left))
                print(compute(gb_right))
                print(compute(gb_result))
                print(np_result)
                print((np_result - compute(gb_result)).new().to_coo()[1])
            assert compare


@pytest.mark.slow
def test_npmonoid():
    values1 = [0, 0, 1, 1, 2, 5]
    values2 = [0, 1, 0, 1, 3, 8]
    index = list(range(len(values1)))
    data = [
        [
            [Vector.from_coo(index, values1), Vector.from_coo(index, values2)],
            [np.array(values1, dtype=int), np.array(values2, dtype=int)],
        ],
        [
            [
                Vector.from_coo(index, values1, dtype="float64"),
                Vector.from_coo(index, values2, dtype="float64"),
            ],
            [np.array(values1, dtype=np.float64), np.array(values2, dtype=np.float64)],
        ],
        [
            [
                Vector.from_coo([0, 1, 2, 3], [True, False, True, False]),
                Vector.from_coo([0, 1, 2, 3], [True, True, False, False]),
            ],
            [np.array([True, False, True, False]), np.array([True, True, False, False])],
        ],
    ]
    # Complex monoids not working yet (they segfault upon creation in gb.core.operators)
    if _supports_complex:
        data.append(
            [
                [
                    Vector.from_coo(index, values1, dtype="FC64"),
                    Vector.from_coo(index, values2, dtype="FC64"),
                ],
                [
                    np.array(values1, dtype=np.complex128),
                    np.array(values2, dtype=np.complex128),
                ],
            ]
        )
    blocklist = {}
    reduction_blocklist = {
        "BOOL": {"add"},
    }
    for (gb_left, gb_right), (np_left, np_right) in data:
        for binary_name in sorted(npmonoid._monoid_identities.keys() & npmonoid.__dir__()):
            op = getattr(npmonoid, binary_name)
            assert len(op.types) > 0, op.name
            if gb_left.dtype not in op.types or binary_name in blocklist.get(
                gb_left.dtype.name, ()
            ):
                continue  # FLAKY COVERAGE
            with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
                gb_result = gb_left.ewise_mult(gb_right, op).new()
                np_result = getattr(np, binary_name)(np_left, np_right)
            np_result = Vector.from_coo(np.arange(np_left.size), np_result, dtype=gb_result.dtype)
            assert gb_result.nvals == np_result.size
            match = gb_result.ewise_mult(np_result, npbinary.equal).new()
            if gb_result.dtype.name.startswith("F"):
                match(accum=gb.binary.lor) << gb_result.apply(npunary.isnan)
            compare = match.reduce(gb.monoid.land).new()
            if not compare:  # pragma: no cover (debug)
                print(binary_name, gb_left.dtype)
                print(compute(gb_result))
                print(np_result)
            assert compare

            # numpy reductions don't have dtype-dependent identities, so results sometimes differ
            if binary_name in reduction_blocklist.get(gb_left.dtype.name, ()):
                continue

            gb_result = gb_left.reduce(op).new()
            np_result = getattr(np, binary_name).reduce(np_left)
            assert gb_result.value == np_result

            gb_result = gb_right.reduce(op).new()
            np_result = getattr(np, binary_name).reduce(np_right)
            assert gb_result.value == np_result


@pytest.mark.slow
def test_npsemiring():
    for monoid_name, binary_name in itertools.product(
        sorted(npmonoid._monoid_identities.keys() & npmonoid.__dir__()),
        sorted(npbinary._binary_names & npbinary.__dir__()),
    ):
        monoid = getattr(npmonoid, monoid_name)
        binary = getattr(npbinary, binary_name)
        name = monoid.name.split(".")[-1] + "_" + binary.name.split(".")[-1]
        if name in {"eq_pow", "eq_minus"}:
            continue
        semiring = gb.core.operator.Semiring.register_anonymous(monoid, binary, name)
        if len(semiring.types) == 0:
            if not gb.config["mapnumpy"] and "logical" not in name:
                assert not hasattr(npsemiring, semiring.name), name
        else:
            assert hasattr(npsemiring, f"{monoid_name}_{binary_name}"), (name, semiring.name)
