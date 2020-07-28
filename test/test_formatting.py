import pytest
import grblas
from grblas import Scalar, Vector, Matrix


@pytest.fixture
def A():
    return Matrix.from_values([0, 0, 0], [0, 2, 4], [0, 1, 2], nrows=1, ncols=5)


@pytest.fixture
def B():
    return Matrix.from_values([0, 2, 4], [0, 0, 0], [10, 20, 30], nrows=5, ncols=1)


@pytest.fixture
def C():
    return Matrix.from_values([0, 9, 60, 69, 0, 9, 60, 69], [4, 4, 4, 4, 72, 72, 72, 72], [0, 2, 3, 4, 5, 6, 7, 8], nrows=70, ncols=77)


@pytest.fixture
def v():
    return Vector.from_values([0, 2, 4], [0, 1, 2])


@pytest.fixture
def w():
    return Vector.from_values([0, 9, 60, 69], [1, 2, 3, 4], size=77)


def test_matrix_repr_small(A, B):
    assert repr(A) == (
        '               nvals  nrows  ncols  dtype\n'
        'grblas.Matrix      3      1      5  INT64\n'
        '-----------------------------------------\n'
        '   0 1  2 3  4\n'
        '0  0    1    2'
    )
    assert repr(B) == (
        '               nvals  nrows  ncols  dtype\n'
        'grblas.Matrix      3      5      1  INT64\n'
        '-----------------------------------------\n'
        '    0\n'
        '0  10\n'
        '1    \n'
        '2  20\n'
        '3    \n'
        '4  30'
    )
    assert repr(B.T) == (
        '                         nvals  nrows  ncols  dtype\n'
        'grblas.TransposedMatrix      3      1      5  INT64\n'
        '---------------------------------------------------\n'
        '    0 1   2 3   4\n'
        '0  10    20    30'
    )


def test_matrix_mask_repr_small(A):
    assert repr(A.S) == (
        '                  nvals  nrows  ncols  dtype\n'
        'StructuralMask  \n'
        'of grblas.Matrix      3      1      5  INT64\n'
        '--------------------------------------------\n'
        '   0 1  2 3  4\n'
        '0  1    1    1'
    )
    assert repr(A.V) == (
        '                  nvals  nrows  ncols  dtype\n'
        'ValueMask       \n'
        'of grblas.Matrix      3      1      5  INT64\n'
        '--------------------------------------------\n'
        '   0 1  2 3  4\n'
        '0  0    1    1'
    )
    assert repr(~A.S) == (
        '                            nvals  nrows  ncols  dtype\n'
        'ComplementedStructuralMask\n'
        'of grblas.Matrix                3      1      5  INT64\n'
        '------------------------------------------------------\n'
        '   0 1  2 3  4\n'
        '0  0    0    0'
    )
    assert repr(~A.V) == (
        '                       nvals  nrows  ncols  dtype\n'
        'ComplementedValueMask\n'
        'of grblas.Matrix           3      1      5  INT64\n'
        '-------------------------------------------------\n'
        '   0 1  2 3  4\n'
        '0  1    0    0'
    )


def test_matrix_repr_large(C):
    assert repr(C) == (
        '               nvals  nrows  ncols  dtype\n'
        'grblas.Matrix      8     70     77  INT64\n'
        '-----------------------------------------\n'
        '   0  1  2  3  4  5  6  7  8  9  10 11  ... 65 66 67 68 69 70 71 72 73 74 75 76\n'
        '0               0                       ...                       5            \n'
        '1                                       ...                                    \n'
        '2                                       ...                                    \n'
        '3                                       ...                                    \n'
        '4                                       ...                                    \n'
        '.. .. .. .. .. .. .. .. .. .. .. .. ..  ... .. .. .. .. .. .. .. .. .. .. .. ..\n'
        '65                                      ...                                    \n'
        '66                                      ...                                    \n'
        '67                                      ...                                    \n'
        '68                                      ...                                    \n'
        '69              4                       ...                       8            '
    )
    assert repr(C.T) == (
        '                         nvals  nrows  ncols  dtype\n'
        'grblas.TransposedMatrix      8     77     70  INT64\n'
        '---------------------------------------------------\n'
        '   0  1  2  3  4  5  6  7  8  9  10 11  ... 58 59 60 61 62 63 64 65 66 67 68 69\n'
        '0                                       ...                                    \n'
        '1                                       ...                                    \n'
        '2                                       ...                                    \n'
        '3                                       ...                                    \n'
        '4   0                          2        ...        3                          4\n'
        '.. .. .. .. .. .. .. .. .. .. .. .. ..  ... .. .. .. .. .. .. .. .. .. .. .. ..\n'
        '72  5                          6        ...        7                          8\n'
        '73                                      ...                                    \n'
        '74                                      ...                                    \n'
        '75                                      ...                                    \n'
        '76                                      ...                                    '
    )


def test_matrix_mask_repr_largs(C):
    assert repr(C.S) == (
        '                  nvals  nrows  ncols  dtype\n'
        'StructuralMask  \n'
        'of grblas.Matrix      8     70     77  INT64\n'
        '--------------------------------------------\n'
        '   0  1  2  3  4  5  6  7  8  9  10 11  ... 65 66 67 68 69 70 71 72 73 74 75 76\n'
        '0               1                       ...                       1            \n'
        '1                                       ...                                    \n'
        '2                                       ...                                    \n'
        '3                                       ...                                    \n'
        '4                                       ...                                    \n'
        '.. .. .. .. .. .. .. .. .. .. .. .. ..  ... .. .. .. .. .. .. .. .. .. .. .. ..\n'
        '65                                      ...                                    \n'
        '66                                      ...                                    \n'
        '67                                      ...                                    \n'
        '68                                      ...                                    \n'
        '69              1                       ...                       1            '
    )
    assert repr(C.V) == (
        '                  nvals  nrows  ncols  dtype\n'
        'ValueMask       \n'
        'of grblas.Matrix      8     70     77  INT64\n'
        '--------------------------------------------\n'
        '   0  1  2  3  4  5  6  7  8  9  10 11  ... 65 66 67 68 69 70 71 72 73 74 75 76\n'
        '0               0                       ...                       1            \n'
        '1                                       ...                                    \n'
        '2                                       ...                                    \n'
        '3                                       ...                                    \n'
        '4                                       ...                                    \n'
        '.. .. .. .. .. .. .. .. .. .. .. .. ..  ... .. .. .. .. .. .. .. .. .. .. .. ..\n'
        '65                                      ...                                    \n'
        '66                                      ...                                    \n'
        '67                                      ...                                    \n'
        '68                                      ...                                    \n'
        '69              1                       ...                       1            '
    )
    assert repr(~C.S) == (
        '                            nvals  nrows  ncols  dtype\n'
        'ComplementedStructuralMask\n'
        'of grblas.Matrix                8     70     77  INT64\n'
        '------------------------------------------------------\n'
        '   0  1  2  3  4  5  6  7  8  9  10 11  ... 65 66 67 68 69 70 71 72 73 74 75 76\n'
        '0               0                       ...                       0            \n'
        '1                                       ...                                    \n'
        '2                                       ...                                    \n'
        '3                                       ...                                    \n'
        '4                                       ...                                    \n'
        '.. .. .. .. .. .. .. .. .. .. .. .. ..  ... .. .. .. .. .. .. .. .. .. .. .. ..\n'
        '65                                      ...                                    \n'
        '66                                      ...                                    \n'
        '67                                      ...                                    \n'
        '68                                      ...                                    \n'
        '69              0                       ...                       0            '
    )
    assert repr(~C.V) == (
        '                       nvals  nrows  ncols  dtype\n'
        'ComplementedValueMask\n'
        'of grblas.Matrix           8     70     77  INT64\n'
        '-------------------------------------------------\n'
        '   0  1  2  3  4  5  6  7  8  9  10 11  ... 65 66 67 68 69 70 71 72 73 74 75 76\n'
        '0               1                       ...                       0            \n'
        '1                                       ...                                    \n'
        '2                                       ...                                    \n'
        '3                                       ...                                    \n'
        '4                                       ...                                    \n'
        '.. .. .. .. .. .. .. .. .. .. .. .. ..  ... .. .. .. .. .. .. .. .. .. .. .. ..\n'
        '65                                      ...                                    \n'
        '66                                      ...                                    \n'
        '67                                      ...                                    \n'
        '68                                      ...                                    \n'
        '69              0                       ...                       0            '
    )
