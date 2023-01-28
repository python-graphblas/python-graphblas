import itertools

import pytest

from graphblas import Vector
from graphblas.core.mask import Mask


@pytest.mark.parametrize("as_matrix", [False, True])
def test_mask_new(as_matrix):
    for dtype, mask_dtype in itertools.product([None, bool, int], [bool, int]):
        v1 = Vector(mask_dtype, size=10)
        v1[3:6] = 0
        v1[:3] = 10
        v2 = Vector(mask_dtype, size=10)
        v2[1::3] = 0
        v2[::3] = 10
        if as_matrix:
            v1 = v1._as_matrix()
            v2 = v2._as_matrix()
        name = "howdy"
        masks = [v1.S, v1.V, ~v1.S, ~v1.V, v2.S, v2.V, ~v2.S, ~v2.V]
        for m1, m2 in itertools.product(masks, masks):
            expected = Vector(bool if dtype is None else dtype, size=10)
            if as_matrix:
                expected = expected._as_matrix()
            expected[...] << True
            expected = expected.dup(mask=m1).dup(mask=m2)
            result = m1.new(dtype, mask=m2, name=name)
            assert result.name == name
            assert result.isequal(expected, check_dtype=True)
            # Complemented
            expected(~expected.S, replace=True) << True
            result = m1.new(dtype, mask=m2, complement=True, name=name)
            assert result.name == name
            assert result.isequal(expected, check_dtype=True)
        # w/o second mask
        for m in masks:
            expected.clear()
            expected[...] << True
            expected = expected.dup(mask=m)
            result = m.new(dtype, name=name)
            assert result.name == name
            assert result.isequal(expected, check_dtype=True)
            # Complemented
            expected(~expected.S, replace=True) << True
            result = m.new(dtype, complement=True, name=name)
            assert result.name == name
            assert result.isequal(expected, check_dtype=True)
        with pytest.raises(TypeError, match="Invalid mask"):
            m.new(mask=object())
        if v1.dtype == bool:
            m.new(mask=v1)  # now okay
        else:
            with pytest.raises(TypeError, match="Mask must be"):
                m.new(mask=v1)


@pytest.mark.parametrize("as_matrix", [False, True])
def test_mask_or(as_matrix):
    for mask_dtype in [bool, int]:
        v1 = Vector(mask_dtype, size=10)
        v1[3:6] = 0
        v1[:3] = 10
        v2 = Vector(mask_dtype, size=10)
        v2[1::3] = 0
        v2[::3] = 10
        if as_matrix:
            v1 = v1._as_matrix()
            v2 = v2._as_matrix()
        masks = [v1.S, v1.V, ~v1.S, ~v1.V, v2.S, v2.V, ~v2.S, ~v2.V]
        for m1, m2 in itertools.product(masks, masks):
            expected = Vector(bool, size=10)
            if as_matrix:
                expected = expected._as_matrix()
            expected(m1) << True
            expected(m2) << True
            result = (m1 | m2).new()
            assert result.isequal(expected, check_dtype=True)
        with pytest.raises(TypeError, match="Invalid mask"):
            m1 | object()
        with pytest.raises(TypeError, match="Invalid mask"):
            object() | m1
        if v1.dtype == bool:
            assert isinstance(m1 | v1, Mask)
            assert isinstance(v1 | m1, Mask)
        else:
            with pytest.raises(TypeError, match="Mask must be"):
                m1 | v1
            with pytest.raises(TypeError, match="Mask must be"):
                v1 | m1


@pytest.mark.parametrize("as_matrix", [False, True])
def test_mask_and(as_matrix):
    for mask_dtype in [bool, int]:
        v1 = Vector(mask_dtype, size=10)
        v1[3:6] = 0
        v1[:3] = 10
        v2 = Vector(mask_dtype, size=10)
        v2[1::3] = 0
        v2[::3] = 10
        if as_matrix:
            v1 = v1._as_matrix()
            v2 = v2._as_matrix()
        masks = [v1.S, v1.V, ~v1.S, ~v1.V, v2.S, v2.V, ~v2.S, ~v2.V]
        for m1, m2 in itertools.product(masks, masks):
            expected = Vector(bool, size=10)
            if as_matrix:
                expected = expected._as_matrix()
            expected[...] << True
            expected = expected.dup(mask=m1).dup(mask=m2)
            result = (m1 & m2).new()
            assert result.isequal(expected, check_dtype=True)
        with pytest.raises(TypeError, match="Invalid mask"):
            m1 & object()
        with pytest.raises(TypeError, match="Invalid mask"):
            object() & m1
        if v1.dtype == bool:
            assert isinstance(m1 & v1, Mask)
            assert isinstance(v1 & m1, Mask)
        else:
            with pytest.raises(TypeError, match="Mask must be"):
                m1 & v1
            with pytest.raises(TypeError, match="Mask must be"):
                v1 & m1
