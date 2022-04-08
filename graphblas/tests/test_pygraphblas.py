import pytest

import graphblas as gb

try:
    import pygraphblas as pg
except ImportError:
    pg = None


@pytest.mark.skipif("not pg")
def test_pygraphblas_matrix():  # pragma: no cover
    if gb.backend != "suitesparse":  # pragma: no cover
        return
    import pygraphblas as pg

    A = gb.Matrix.from_values([0, 1], [0, 1], [0, 1])
    pgA = A.to_pygraphblas()
    assert isinstance(pgA, pg.Matrix)
    assert A.gb_obj is gb.ffi.NULL
    pgA += 1
    A2 = gb.Matrix.from_pygraphblas(pgA)
    assert A2.isequal(gb.Matrix.from_values([0, 1], [0, 1], [1, 2]))
    assert A.dtype == A2.dtype


@pytest.mark.skipif("not pg")
def test_pygraphblas_vector():  # pragma: no cover
    if gb.backend != "suitesparse":  # pragma: no cover
        return
    import pygraphblas as pg

    v = gb.Vector.from_values([0, 2], [0, 1])
    pgv = v.to_pygraphblas()
    assert isinstance(pgv, pg.Vector)
    assert v.gb_obj is gb.ffi.NULL
    pgv += 1
    v2 = gb.Vector.from_pygraphblas(pgv)
    assert v2.isequal(gb.Vector.from_values([0, 2], [1, 2]))
    assert v.dtype == v2.dtype


@pytest.mark.skipif("not pg")
def test_pygraphblas_scalar():  # pragma: no cover
    if gb.backend != "suitesparse":  # pragma: no cover
        return
    import pygraphblas as pg

    s = gb.Scalar.from_value(1)
    pgs = s.to_pygraphblas()
    assert isinstance(pgs, pg.Scalar)
    assert pgs.nvals == 1
    s2 = gb.Scalar.from_pygraphblas(pgs)
    assert s2 == 1
    assert s.dtype == s2.dtype
