import pytest
import grblas
from grblas import Matrix, Vector
from grblas import unary, binary
from grblas import dtypes
from grblas.base import Updater


def test_from_values_dtype_resolving():
    u = Vector.from_values([0, 1, 2], [1, 2, 3], dtype=dtypes.INT32)
    assert u.dtype == 'INT32'
    u = Vector.from_values([0, 1, 2], [1, 2, 3], dtype='INT32')
    assert u.dtype == dtypes.INT32
    M = Matrix.from_values([0, 1, 2], [2, 0, 1], [0, 2, 3], dtype=dtypes.UINT8)
    assert M.dtype == 'UINT8'
    M = Matrix.from_values([0, 1, 2], [2, 0, 1], [0, 2, 3], dtype=float)
    assert M.dtype == dtypes.FP64


def test_from_values_invalid_dtype():
    with pytest.raises(OverflowError):
        Matrix.from_values([0, 1, 2], [2, 0, 1], [0, 2, 3], dtype=dtypes.BOOL)


def test_resolve_ops_using_common_dtype():
    # C << A.ewise_mult(B, binary.plus) <-- PLUS should use FP64 because unify(INT64, FP64) -> FP64
    u = Vector.from_values([0, 1, 3], [1, 2, 3], dtype=dtypes.INT64)
    v = Vector.from_values([0, 1, 3], [0.1, 0.1, 0.1], dtype='FP64')
    w = Vector.new('FP32', u.size)
    w << u.ewise_mult(v, binary.plus)
    result = Vector.from_values([0, 1, 3], [1.1, 2.1, 3.1], dtype='FP32')
    assert w.isclose(result, check_dtype=True)


def test_order_of_updater_params_does_not_matter():
    u = Vector.from_values([0, 1, 3], [1, 2, 3])
    mask = Vector.from_values([0, 3], [True, True])
    accum = binary.plus
    result = Vector.from_values([0, 3], [5, 10])
    # mask, accum, replace=
    v = Vector.from_values([0, 1, 2, 3], [4, 3, 2, 1])
    v(mask.V, accum, replace=True) << u.ewise_mult(u, binary.times)
    assert v.isequal(result)
    # accum, mask, replace=
    v = Vector.from_values([0, 1, 2, 3], [4, 3, 2, 1])
    v(accum, mask.V, replace=True) << u.ewise_mult(u, binary.times)
    assert v.isequal(result)
    # accum, mask=, replace=
    v = Vector.from_values([0, 1, 2, 3], [4, 3, 2, 1])
    v(accum, mask=mask.V, replace=True) << u.ewise_mult(u, binary.times)
    assert v.isequal(result)
    # mask, accum=, replace=
    v = Vector.from_values([0, 1, 2, 3], [4, 3, 2, 1])
    v(mask.V, accum=accum, replace=True) << u.ewise_mult(u, binary.times)
    assert v.isequal(result)
    # replace=, mask=, accum=
    v = Vector.from_values([0, 1, 2, 3], [4, 3, 2, 1])
    v(replace=True, mask=mask.V, accum=accum) << u.ewise_mult(u, binary.times)
    assert v.isequal(result)


def test_already_resolved_ops_allowed_in_updater():
    # C(binary.plus['FP64']) << ...
    u = Vector.from_values([0, 1, 3], [1, 2, 3])
    u(binary.plus['INT64']) << u.ewise_mult(u, binary.times['INT64'])
    result = Vector.from_values([0, 1, 3], [2, 6, 12])
    assert u.isequal(result)


def test_updater_returns_updater():
    u = Vector.from_values([0, 1, 3], [1, 2, 3])
    y = u(accum=binary.times)
    assert isinstance(y, Updater)
    z = y << u.apply(unary.ainv)
    assert z is None
    assert isinstance(y, Updater)
    final_result = Vector.from_values([0, 1, 3], [-1, -4, -9])
    assert u.isequal(final_result)


def test_updater_only_once():
    u = Vector.from_values([0, 1, 3], [1, 2, 3])
    with pytest.raises(ValueError, match="already called.*no keywords"):
        u()[0]()
    with pytest.raises(ValueError, match="already called.*mask="):
        u(mask=u.S)[0]()
    with pytest.raises(ValueError, match="already called.*accum="):
        u(accum=binary.plus)[0]()
    with pytest.raises(TypeError, match="not callable"):
        u()()
    with pytest.raises(ValueError, match="already called.*no keywords"):
        u[[0, 1]]()()
    # While we're at it...
    with pytest.raises(TypeError, match="is not subscriptable"):
        u[[0, 1]]()[0]
    with pytest.raises(TypeError, match="is not subscriptable"):
        u()[[0, 1]][0]
    with pytest.raises(TypeError, match="is not subscriptable"):
        u[[0, 1]][0]


def test_import_special_attrs():
    not_hidden = {x for x in dir(grblas) if not x.startswith('_')}
    # Is everything imported?
    assert len(not_hidden & grblas._SPECIAL_ATTRS) == len(grblas._SPECIAL_ATTRS)
    # Is everything special that needs to be?
    not_special = {x for x in dir(grblas) if not x.startswith('_')} - grblas._SPECIAL_ATTRS
    assert not_special == {'backends', 'init', 'mask'}
    # Make sure these "not special" objects don't have objects that look special within them
    for attr in not_special:
        assert not set(dir(getattr(grblas, attr))) & grblas._SPECIAL_ATTRS
