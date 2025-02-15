import itertools

import pytest

import graphblas as gb
from graphblas import Vector
from graphblas.core.mask import Mask

from .conftest import burble, dprint


@pytest.mark.parametrize("as_matrix", [False, True])
def test_mask_new(as_matrix):
    for dtype, mask_dtype in itertools.product([None, bool, int], [bool, int]):
        # debug print used to investigate segfaults
        dprint("G", 0)
        v1 = Vector(mask_dtype, size=10)
        dprint("G", 1)
        v1[3:6] = 0
        dprint("G", 2)
        v1[:3] = 10
        dprint("G", 3)
        v2 = Vector(mask_dtype, size=10)
        dprint("G", 4)
        v2[1::3] = 0
        dprint("G", 5)
        v2[::3] = 10
        dprint("G", 6)
        if as_matrix:
            v1 = v1._as_matrix()
            dprint("G", 7)
            v2 = v2._as_matrix()
            dprint("G", 8)
        name = "howdy"
        masks = [v1.S, v1.V, ~v1.S, ~v1.V, v2.S, v2.V, ~v2.S, ~v2.V]
        dprint("G", 9)
        for m1, m2 in itertools.product(masks, masks):
            expected = Vector(bool if dtype is None else dtype, size=10)
            dprint("G", 10)
            if as_matrix:
                expected = expected._as_matrix()
                dprint("G", 11)
            expected[...] << True
            dprint("G", 12)
            expected = expected.dup(mask=m1).dup(mask=m2)
            dprint("G", 13)
            dprint("(x_x;)")
            dprint(m1)
            dprint(m2)
            with gb.Recorder(), burble():
                dprint("Recorded. About to crash!")
                result = m1.new(dtype, mask=m2, name=name)  # XXX: here
            dprint("G", 14)
            assert result.name == name
            dprint("G", 15)
            assert result.isequal(expected, check_dtype=True)
            dprint("G", 16)
            # Complemented
            expected(~expected.S, replace=True) << True
            dprint("G", 17)
            result = m1.new(dtype, mask=m2, complement=True, name=name)
            dprint("G", 18)
            assert result.name == name
            dprint("G", 19)
            assert result.isequal(expected, check_dtype=True)
            dprint("G", 20)
        # w/o second mask
        for m in masks:
            expected.clear()
            dprint("G", 21)
            expected[...] << True
            dprint("G", 22)
            expected = expected.dup(mask=m)
            dprint("G", 23)
            result = m.new(dtype, name=name)
            dprint("G", 24)
            assert result.name == name
            dprint("G", 25)
            assert result.isequal(expected, check_dtype=True)
            dprint("G", 26)
            # Complemented
            expected(~expected.S, replace=True) << True
            dprint("G", 27)
            result = m.new(dtype, complement=True, name=name)
            dprint("G", 28)
            assert result.name == name
            dprint("G", 29)
            assert result.isequal(expected, check_dtype=True)
            dprint("G", 30)
        with pytest.raises(TypeError, match="Invalid mask"):
            m.new(mask=object())
        dprint("G", 31)
        if v1.dtype == bool:
            m.new(mask=v1)  # now okay
            dprint("G", 32)
        else:
            with pytest.raises(TypeError, match="Mask must be"):
                m.new(mask=v1)
            dprint("G", 33)


@pytest.mark.parametrize("as_matrix", [False, True])
def test_mask_or(as_matrix):
    for mask_dtype in [bool, int]:
        # debug print used to investigate segfaults
        dprint("H", 0)
        v1 = Vector(mask_dtype, size=10)
        dprint("H", 1)
        v1[3:6] = 0
        dprint("H", 2)
        v1[:3] = 10
        dprint("H", 3)
        v2 = Vector(mask_dtype, size=10)
        dprint("H", 4)
        v2[1::3] = 0
        dprint("H", 5)
        v2[::3] = 10
        dprint("H", 6)
        if as_matrix:
            v1 = v1._as_matrix()
            dprint("H", 7)
            v2 = v2._as_matrix()
            dprint("H", 8)
        masks = [v1.S, v1.V, ~v1.S, ~v1.V, v2.S, v2.V, ~v2.S, ~v2.V]
        dprint("H", 9)
        for m1, m2 in itertools.product(masks, masks):
            expected = Vector(bool, size=10)
            dprint("H", 10)
            if as_matrix:
                expected = expected._as_matrix()
                dprint("H", 11)
            expected(m1) << True
            dprint("H", 12)
            expected(m2) << True
            dprint("H", 13)
            dprint("(x_x;)")
            dprint(m1)
            dprint(m2)
            with gb.Recorder(), burble():
                dprint("Recorded. About to crash!", mask_dtype)
                result = (m1 | m2).new()  # XXX: here
            dprint("H", 14)
            assert result.isequal(expected, check_dtype=True)
            dprint("H", 15)
        with pytest.raises(TypeError, match="Invalid mask"):
            m1 | object()
        dprint("H", 16)
        with pytest.raises(TypeError, match="Invalid mask"):
            object() | m1
        dprint("H", 17)
        if v1.dtype == bool:
            assert isinstance(m1 | v1, Mask)
            dprint("H", 18)
            assert isinstance(v1 | m1, Mask)
            dprint("H", 19)
        else:
            with pytest.raises(TypeError, match="Mask must be"):
                m1 | v1
            dprint("H", 20)
            with pytest.raises(TypeError, match="Mask must be"):
                v1 | m1
            dprint("H", 21)


@pytest.mark.parametrize("as_matrix", [False, True])
def test_mask_and(as_matrix):
    for mask_dtype in [bool, int]:
        # debug print used to investigate segfaults
        dprint("I", 0)
        v1 = Vector(mask_dtype, size=10)
        dprint("I", 1)
        v1[3:6] = 0
        dprint("I", 2)
        v1[:3] = 10
        dprint("I", 3)
        v2 = Vector(mask_dtype, size=10)
        dprint("I", 4)
        v2[1::3] = 0
        dprint("I", 5)
        v2[::3] = 10
        dprint("I", 6)
        if as_matrix:
            v1 = v1._as_matrix()
            dprint("I", 7)
            v2 = v2._as_matrix()
            dprint("I", 8)
        masks = [v1.S, v1.V, ~v1.S, ~v1.V, v2.S, v2.V, ~v2.S, ~v2.V]
        dprint("I", 9)
        for m1, m2 in itertools.product(masks, masks):
            expected = Vector(bool, size=10)
            dprint("I", 10)
            if as_matrix:
                expected = expected._as_matrix()
                dprint("I", 11)
            expected[...] << True
            dprint("I", 12)
            expected = expected.dup(mask=m1).dup(mask=m2)
            dprint("I", 13)
            dprint("(x_x;)")
            dprint(m1)
            dprint(m2)
            with gb.Recorder(), burble():
                dprint("Recorded. About to crash!")
                result = (m1 & m2).new()  # XXX: here
            dprint("I", 14)
            assert result.isequal(expected, check_dtype=True)
            dprint("I", 15)
        with pytest.raises(TypeError, match="Invalid mask"):
            m1 & object()
        dprint("I", 16)
        with pytest.raises(TypeError, match="Invalid mask"):
            object() & m1
        dprint("I", 17)
        if v1.dtype == bool:
            assert isinstance(m1 & v1, Mask)
            dprint("I", 18)
            assert isinstance(v1 & m1, Mask)
            dprint("I", 19)
        else:
            with pytest.raises(TypeError, match="Mask must be"):
                m1 & v1
            dprint("I", 20)
            with pytest.raises(TypeError, match="Mask must be"):
                v1 & m1
            dprint("I", 21)
