import itertools

from pytest import raises

from graphblas import Vector


def test_mask_new():
    for dtype, mask_dtype in itertools.product([None, bool, int], [bool, int]):
        v1 = Vector(mask_dtype, size=10)
        v1[3:6] = 0
        v1[:3] = 10
        v2 = Vector(mask_dtype, size=10)
        v2[1::3] = 0
        v2[::3] = 10
        name = "howdy"
        masks = [v1.S, v1.V, ~v1.S, ~v1.V, v2.S, v2.V, ~v2.S, ~v2.V]
        for m1, m2 in itertools.product(masks, masks):
            expected = Vector(bool if dtype is None else dtype, size=v1.size)
            expected << True
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
            expected << True
            expected = expected.dup(mask=m)
            result = m.new(dtype, name=name)
            assert result.name == name
            assert result.isequal(expected, check_dtype=True)
            # Complemented
            expected(~expected.S, replace=True) << True
            result = m.new(dtype, complement=True, name=name)
            assert result.name == name
            assert result.isequal(expected, check_dtype=True)
        with raises(TypeError, match="Invalid mask"):
            m.new(mask=object())
        with raises(TypeError, match="Mask must indicate"):
            m.new(mask=v1)
