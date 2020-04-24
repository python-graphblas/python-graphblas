# These tests are very slow, since they force creation of all numpy unary, binary, monoid, and semiring objects
import pytest
import numpy as np
import itertools
import grblas
import grblas.unary.numpy as npunary
import grblas.binary.numpy as npbinary
import grblas.monoid.numpy as npmonoid
import grblas.semiring.numpy as npsemiring


def test_bool_doesnt_get_too_large():
    a = grblas.Vector.from_values([0, 1, 2, 3], [True, False, True, False])
    b = grblas.Vector.from_values([0, 1, 2, 3], [True, True, False, False])
    z = a.ewise_mult(b, grblas.monoid.numpy.add).new()
    z.show()
    x, y = z.to_values()
    assert y == (True, True, True, False)

    op = grblas.ops.UnaryOp.register_anonymous(lambda x: np.add(x, x))
    z = a.apply(op).new()
    z.show()
    x, y = z.to_values()
    assert y == (True, False, True, False)


@pytest.mark.slow
def test_npunary():
    L = list(range(5))
    data = [
        [grblas.Vector.from_values([0, 1], [True, False]), np.array([True, False])],
        [grblas.Vector.from_values(L, L), np.array(L, dtype=int)],
        [grblas.Vector.from_values(L, L, dtype='float64'), np.array(L, dtype=np.float64)],
    ]
    blacklist = {}
    isclose = grblas.vector._generate_isclose(1e-7, 0)
    for gb_input, np_input in data:
        for unary_name in sorted(npunary._unary_names):
            op = getattr(npunary, unary_name)
            if gb_input.dtype.name not in op.types or unary_name in blacklist.get(gb_input.dtype.name, ()):
                continue
            with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
                gb_result = gb_input.apply(op).new()
                if gb_input.dtype == 'BOOL' and gb_result.dtype == 'FP32':
                    np_result = getattr(np, unary_name)(np_input, dtype='float32')
                    compare_op = isclose
                else:
                    np_result = getattr(np, unary_name)(np_input)
                    compare_op = npbinary.equal
            np_result = grblas.Vector.from_values(list(range(np_input.size)), list(np_result), dtype=gb_result.dtype)
            assert gb_result.nvals == np_result.size
            match = gb_result.ewise_mult(np_result, compare_op).new()
            match(accum=grblas.binary.lor) << gb_result.apply(npunary.isnan)
            compare = match.reduce(grblas.monoid.land).value
            if not compare:
                print(unary_name, gb_input.dtype)
                print(gb_result.show())
                print(np_result.show())
            assert compare


@pytest.mark.slow
def test_npbinary():
    values1 = [0, 0, 1, 1, 2, 5]
    values2 = [0, 1, 0, 1, 3, 8]
    index = list(range(len(values1)))
    data = [
        [
            [
                grblas.Vector.from_values(index, values1),
                grblas.Vector.from_values(index, values2),
            ],
            [
                np.array(values1, dtype=int),
                np.array(values2, dtype=int),
            ],
        ],
        [
            [
                grblas.Vector.from_values(index, values1, dtype='float64'),
                grblas.Vector.from_values(index, values2, dtype='float64'),
            ],
            [
                np.array(values1, dtype=np.float64),
                np.array(values2, dtype=np.float64),
            ],
        ],
        [
            [
                grblas.Vector.from_values([0, 1, 2, 3], [True, False, True, False]),
                grblas.Vector.from_values([0, 1, 2, 3], [True, True, False, False]),
            ],
            [
                np.array([True, False, True, False]),
                np.array([True, True, False, False]),
            ],
        ],
    ]
    blacklist = {
        'FP64': {
            'floor_divide',  # numba/numpy difference for 1.0 / 0.0
        },
    }
    isclose = grblas.vector._generate_isclose(1e-7, 0)
    for (gb_left, gb_right), (np_left, np_right) in data:
        for binary_name in sorted(npbinary._binary_names):
            op = getattr(npbinary, binary_name)
            if gb_left.dtype.name not in op.types or binary_name in blacklist.get(gb_left.dtype.name, ()):
                continue
            with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
                gb_result = gb_left.ewise_mult(gb_right, op).new()
                if gb_left.dtype == 'BOOL' and gb_result.dtype == 'FP32':
                    np_result = getattr(np, binary_name)(np_left, np_right, dtype='float32')
                    compare_op = isclose
                else:
                    np_result = getattr(np, binary_name)(np_left, np_right)
                    compare_op = npbinary.equal
            np_result = grblas.Vector.from_values(list(range(np_left.size)), list(np_result), dtype=gb_result.dtype)
            assert gb_result.nvals == np_result.size
            match = gb_result.ewise_mult(np_result, compare_op).new()
            match(accum=grblas.binary.lor) << gb_result.apply(npunary.isnan)
            compare = match.reduce(grblas.monoid.land).value
            if not compare:
                print(binary_name, gb_left.dtype)
                print(gb_result.show())
                print(np_result.show())
            assert compare


@pytest.mark.slow
def test_npmonoid():
    values1 = [0, 0, 1, 1, 2, 5]
    values2 = [0, 1, 0, 1, 3, 8]
    index = list(range(len(values1)))
    data = [
        [
            [
                grblas.Vector.from_values(index, values1),
                grblas.Vector.from_values(index, values2),
            ],
            [
                np.array(values1, dtype=int),
                np.array(values2, dtype=int),
            ],
        ],
        [
            [
                grblas.Vector.from_values(index, values1, dtype='float64'),
                grblas.Vector.from_values(index, values2, dtype='float64'),
            ],
            [
                np.array(values1, dtype=np.float64),
                np.array(values2, dtype=np.float64),
            ],
        ],
        [
            [
                grblas.Vector.from_values([0, 1, 2, 3], [True, False, True, False]),
                grblas.Vector.from_values([0, 1, 2, 3], [True, True, False, False]),
            ],
            [
                np.array([True, False, True, False]),
                np.array([True, True, False, False]),
            ],
        ],
    ]
    blacklist = {}
    reduction_blacklist = {
        'BOOL': {'add'},
    }
    for (gb_left, gb_right), (np_left, np_right) in data:
        for binary_name in sorted(npmonoid._monoid_identities):
            op = getattr(npmonoid, binary_name)
            assert len(op.types) > 0, op.name
            if gb_left.dtype.name not in op.types or binary_name in blacklist.get(gb_left.dtype.name, ()):
                continue
            with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
                gb_result = gb_left.ewise_mult(gb_right, op).new()
                np_result = getattr(np, binary_name)(np_left, np_right)
            np_result = grblas.Vector.from_values(list(range(np_left.size)), list(np_result), dtype=gb_result.dtype)
            assert gb_result.nvals == np_result.size
            match = gb_result.ewise_mult(np_result, npbinary.equal).new()
            match(accum=grblas.binary.lor) << gb_result.apply(npunary.isnan)
            compare = match.reduce(grblas.monoid.land).value
            if not compare:
                print(binary_name, gb_left.dtype)
                print(gb_result.show())
                print(np_result.show())
            assert compare

            # numpy reductions don't have dtype-dependent identities, so results sometimes differ
            if binary_name in reduction_blacklist.get(gb_left.dtype.name, ()):
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
        sorted(npmonoid._monoid_identities),
        sorted(npbinary._binary_names)
    ):
        monoid = getattr(npmonoid, monoid_name)
        binary = getattr(npbinary, binary_name)
        name = monoid.name.split(".")[-1] + "_" + binary.name.split(".")[-1]
        semiring = grblas.ops.Semiring.register_anonymous(monoid, binary, name)
        if len(semiring.types) == 0:
            assert not hasattr(npsemiring, semiring.name), name
        else:
            assert hasattr(npsemiring, semiring.name), name
