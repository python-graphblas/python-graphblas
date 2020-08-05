import pytest
import pandas as pd
import grblas
from grblas import Scalar, Vector, Matrix, formatting, unary


def repr_html(x):
    return x._repr_html_()


def _printer(text, name, repr_name, indent):
    indent = " " * indent
    print(f"{indent}assert {repr_name}({name}) == (")
    lines = text.split("\n")
    prev_line = ""
    count = 0
    for line in lines[:-1]:
        line = line + "\n"
        if line == prev_line:
            count += 1
        else:
            if count == 1:
                print(f"{indent}    {prev_line!r}")
            elif count > 1:
                print(f"{indent}  + {prev_line!r} * {count} +")
            count = 1
            prev_line = line
    if count == 1:
        print(f"{indent}    {prev_line!r}")
    elif count > 1:  # pragma: no cover
        print(f"{indent}    {prev_line!r} * {count}")
    print(f"{indent}    {lines[-1]!r}")
    print(f"{indent})")


def repr_printer(x, name="", indent=4):
    return _printer(repr(x), name, "repr", indent)


def html_printer(x, name="", indent=4):
    return _printer(repr_html(x), name, "repr_html", indent)


@pytest.fixture
def A():
    return Matrix.from_values([0, 0, 0], [0, 2, 4], [0, 1, 2], nrows=1, ncols=5, name="A_1")


@pytest.fixture
def B():
    return Matrix.from_values([0, 2, 4], [0, 0, 0], [10, 20, 30], nrows=5, ncols=1, name="B_1")


@pytest.fixture
def C():
    return Matrix.from_values(
        [0, 9, 60, 69, 0, 9, 60, 69],
        [4, 4, 4, 4, 72, 72, 72, 72],
        [0, 2, 3, 4, 5, 6, 7, 8],
        nrows=70,
        ncols=77,
        name="C",
    )


@pytest.fixture
def D():
    return Matrix.from_values(
        [0, 9, 60, 69],
        [4, 4, 4, 4],
        [True, False, True, False],
        nrows=70,
        name="D_skinny_in_one_dim",
    )


@pytest.fixture
def v():
    return Vector.from_values([0, 2, 4], [0.0, 1.1, 2.2], name="v")


@pytest.fixture
def w():
    return Vector.from_values([0, 5, 64, 69], [1, 2, 3, 4], size=77, name="w")


@pytest.fixture
def s():
    return Scalar.from_value(42, name="s_1")


@pytest.fixture
def t():
    return Scalar.new(int, name="t")


def test_no_pandas_repr(A, C, v, w):
    # This is a bit of a hack...
    formatting.has_pandas = False
    try:
        repr_printer(A, "A", indent=8)
        assert repr(A) == (
            '"A_1"          nvals  nrows  ncols  dtype\n'
            "grblas.Matrix      3      1      5  INT64"
        )
        repr_printer(A.T, "A.T", indent=8)
        assert repr(A.T) == (
            '"A_1.T"                  nvals  nrows  ncols  dtype\n'
            "grblas.TransposedMatrix      3      5      1  INT64"
        )
        repr_printer(C.S, "C.S", indent=8)
        assert repr(C.S) == (
            '"C.S"             nvals  nrows  ncols  dtype\n'
            "StructuralMask  \n"
            "of grblas.Matrix      8     70     77  INT64"
        )
        repr_printer(v, "v", indent=8)
        assert repr(v) == (
            '"v"            nvals  size  dtype\n' "grblas.Vector      3     5   FP64"
        )
        repr_printer(~w.V, "~w.V", indent=8)
        assert repr(~w.V) == (
            '"~w.V"                 nvals  size  dtype\n'
            "ComplementedValueMask\n"
            "of grblas.Vector           4    77  INT64"
        )
    finally:
        formatting.has_pandas = True


def test_matrix_repr_small(A, B):
    repr_printer(A, "A")
    assert repr(A) == (
        '"A_1"          nvals  nrows  ncols  dtype\n'
        "grblas.Matrix      3      1      5  INT64\n"
        "-----------------------------------------\n"
        "   0 1  2 3  4\n"
        "0  0    1    2"
    )
    repr_printer(B, "B")
    assert repr(B) == (
        '"B_1"          nvals  nrows  ncols  dtype\n'
        "grblas.Matrix      3      5      1  INT64\n"
        "-----------------------------------------\n"
        "    0\n"
        "0  10\n"
        "1    \n"
        "2  20\n"
        "3    \n"
        "4  30"
    )
    repr_printer(B.T, "B.T")
    assert repr(B.T) == (
        '"B_1.T"                  nvals  nrows  ncols  dtype\n'
        "grblas.TransposedMatrix      3      1      5  INT64\n"
        "---------------------------------------------------\n"
        "    0 1   2 3   4\n"
        "0  10    20    30"
    )


def test_matrix_mask_repr_small(A):
    repr_printer(A.S, "A.S")
    assert repr(A.S) == (
        '"A_1.S"           nvals  nrows  ncols  dtype\n'
        "StructuralMask  \n"
        "of grblas.Matrix      3      1      5  INT64\n"
        "--------------------------------------------\n"
        "   0 1  2 3  4\n"
        "0  1    1    1"
    )
    repr_printer(A.V, "A.V")
    assert repr(A.V) == (
        '"A_1.V"           nvals  nrows  ncols  dtype\n'
        "ValueMask       \n"
        "of grblas.Matrix      3      1      5  INT64\n"
        "--------------------------------------------\n"
        "   0 1  2 3  4\n"
        "0  0    1    1"
    )
    repr_printer(~A.S, "~A.S")
    assert repr(~A.S) == (
        '"~A_1.S"                    nvals  nrows  ncols  dtype\n'
        "ComplementedStructuralMask\n"
        "of grblas.Matrix                3      1      5  INT64\n"
        "------------------------------------------------------\n"
        "   0 1  2 3  4\n"
        "0  0    0    0"
    )
    repr_printer(~A.V, "~A.V")
    assert repr(~A.V) == (
        '"~A_1.V"               nvals  nrows  ncols  dtype\n'
        "ComplementedValueMask\n"
        "of grblas.Matrix           3      1      5  INT64\n"
        "-------------------------------------------------\n"
        "   0 1  2 3  4\n"
        "0  1    0    0"
    )


def test_matrix_repr_large(C, D):
    with pd.option_context("display.max_columns", 24, "display.width", 100):
        repr_printer(C, "C", indent=8)
        assert repr(C) == (
            '"C"            nvals  nrows  ncols  dtype\n'
            "grblas.Matrix      8     70     77  INT64\n"
            "-----------------------------------------\n"
            "   0  1  2  3  4  5  6  7  8  9  10 11  ... 65 66 67 68 69 70 71 72 73 74 75 76\n"
            "0               0                       ...                       5            \n"
            "1                                       ...                                    \n"
            "2                                       ...                                    \n"
            "3                                       ...                                    \n"
            "4                                       ...                                    \n"
            ".. .. .. .. .. .. .. .. .. .. .. .. ..  ... .. .. .. .. .. .. .. .. .. .. .. ..\n"
            "65                                      ...                                    \n"
            "66                                      ...                                    \n"
            "67                                      ...                                    \n"
            "68                                      ...                                    \n"
            "69              4                       ...                       8            "
        )
        repr_printer(C.T, "C.T", indent=8)
        assert repr(C.T) == (
            '"C.T"                    nvals  nrows  ncols  dtype\n'
            "grblas.TransposedMatrix      8     77     70  INT64\n"
            "---------------------------------------------------\n"
            "   0  1  2  3  4  5  6  7  8  9  10 11  ... 58 59 60 61 62 63 64 65 66 67 68 69\n"
            "0                                       ...                                    \n"
            "1                                       ...                                    \n"
            "2                                       ...                                    \n"
            "3                                       ...                                    \n"
            "4   0                          2        ...        3                          4\n"
            ".. .. .. .. .. .. .. .. .. .. .. .. ..  ... .. .. .. .. .. .. .. .. .. .. .. ..\n"
            "72  5                          6        ...        7                          8\n"
            "73                                      ...                                    \n"
            "74                                      ...                                    \n"
            "75                                      ...                                    \n"
            "76                                      ...                                    "
        )
        repr_printer(D, "D", indent=8)
        assert repr(D) == (
            '"D_skinny_in_one_dim"  nvals  nrows  ncols  dtype\n'
            "grblas.Matrix              4     70      5   BOOL\n"
            "-------------------------------------------------\n"
            "   0  1  2  3       4\n"
            "0                True\n"
            "1                    \n"
            "2                    \n"
            "3                    \n"
            "4                    \n"
            ".. .. .. .. ..    ...\n"
            "65                   \n"
            "66                   \n"
            "67                   \n"
            "68                   \n"
            "69              False"
        )
        repr_printer(D.T, "D.T", indent=8)
        assert repr(D.T) == (
            '"D_skinny_in_one_dim.T"  nvals  nrows  ncols  dtype\n'
            "grblas.TransposedMatrix      4      5     70   BOOL\n"
            "---------------------------------------------------\n"
            "     0  1  2  3  4  5  6  7  8      9  10 11  ... 58 59    60 61 62 63 64 65 66 67 68     69\n"
            "0                                             ...                                           \n"
            "1                                             ...                                           \n"
            "2                                             ...                                           \n"
            "3                                             ...                                           \n"
            "4  True                          False        ...        True                          False"
        )


def test_matrix_mask_repr_large(C):
    with pd.option_context("display.max_columns", 24, "display.width", 100):
        repr_printer(C.S, "C.S", indent=8)
        assert repr(C.S) == (
            '"C.S"             nvals  nrows  ncols  dtype\n'
            "StructuralMask  \n"
            "of grblas.Matrix      8     70     77  INT64\n"
            "--------------------------------------------\n"
            "   0  1  2  3  4  5  6  7  8  9  10 11  ... 65 66 67 68 69 70 71 72 73 74 75 76\n"
            "0               1                       ...                       1            \n"
            "1                                       ...                                    \n"
            "2                                       ...                                    \n"
            "3                                       ...                                    \n"
            "4                                       ...                                    \n"
            ".. .. .. .. .. .. .. .. .. .. .. .. ..  ... .. .. .. .. .. .. .. .. .. .. .. ..\n"
            "65                                      ...                                    \n"
            "66                                      ...                                    \n"
            "67                                      ...                                    \n"
            "68                                      ...                                    \n"
            "69              1                       ...                       1            "
        )
        repr_printer(C.V, "C.V", indent=8)
        assert repr(C.V) == (
            '"C.V"             nvals  nrows  ncols  dtype\n'
            "ValueMask       \n"
            "of grblas.Matrix      8     70     77  INT64\n"
            "--------------------------------------------\n"
            "   0  1  2  3  4  5  6  7  8  9  10 11  ... 65 66 67 68 69 70 71 72 73 74 75 76\n"
            "0               0                       ...                       1            \n"
            "1                                       ...                                    \n"
            "2                                       ...                                    \n"
            "3                                       ...                                    \n"
            "4                                       ...                                    \n"
            ".. .. .. .. .. .. .. .. .. .. .. .. ..  ... .. .. .. .. .. .. .. .. .. .. .. ..\n"
            "65                                      ...                                    \n"
            "66                                      ...                                    \n"
            "67                                      ...                                    \n"
            "68                                      ...                                    \n"
            "69              1                       ...                       1            "
        )
        repr_printer(~C.S, "~C.S", indent=8)
        assert repr(~C.S) == (
            '"~C.S"                      nvals  nrows  ncols  dtype\n'
            "ComplementedStructuralMask\n"
            "of grblas.Matrix                8     70     77  INT64\n"
            "------------------------------------------------------\n"
            "   0  1  2  3  4  5  6  7  8  9  10 11  ... 65 66 67 68 69 70 71 72 73 74 75 76\n"
            "0               0                       ...                       0            \n"
            "1                                       ...                                    \n"
            "2                                       ...                                    \n"
            "3                                       ...                                    \n"
            "4                                       ...                                    \n"
            ".. .. .. .. .. .. .. .. .. .. .. .. ..  ... .. .. .. .. .. .. .. .. .. .. .. ..\n"
            "65                                      ...                                    \n"
            "66                                      ...                                    \n"
            "67                                      ...                                    \n"
            "68                                      ...                                    \n"
            "69              0                       ...                       0            "
        )
        repr_printer(~C.V, "~C.V", indent=8)
        assert repr(~C.V) == (
            '"~C.V"                 nvals  nrows  ncols  dtype\n'
            "ComplementedValueMask\n"
            "of grblas.Matrix           8     70     77  INT64\n"
            "-------------------------------------------------\n"
            "   0  1  2  3  4  5  6  7  8  9  10 11  ... 65 66 67 68 69 70 71 72 73 74 75 76\n"
            "0               1                       ...                       0            \n"
            "1                                       ...                                    \n"
            "2                                       ...                                    \n"
            "3                                       ...                                    \n"
            "4                                       ...                                    \n"
            ".. .. .. .. .. .. .. .. .. .. .. .. ..  ... .. .. .. .. .. .. .. .. .. .. .. ..\n"
            "65                                      ...                                    \n"
            "66                                      ...                                    \n"
            "67                                      ...                                    \n"
            "68                                      ...                                    \n"
            "69              0                       ...                       0            "
        )


def test_vector_repr_small(v):
    repr_printer(v, "v")
    assert repr(v) == (
        '"v"            nvals  size  dtype\n'
        "grblas.Vector      3     5   FP64\n"
        "---------------------------------\n"
        "  0 1    2 3    4\n"
        "  0    1.1    2.2"
    )


def test_vector_repr_large(w):
    with pd.option_context("display.max_columns", 26, "display.width", 100):
        repr_printer(w, "w", indent=8)
        assert repr(w) == (
            '"w"            nvals  size  dtype\n'
            "grblas.Vector      4    77  INT64\n"
            "---------------------------------\n"
            " 0  1  2  3  4  5  6  7  8  9  10 11 12  ... 64 65 66 67 68 69 70 71 72 73 74 75 76\n"
            "  1              2                       ...  3              4                     "
        )


def test_vector_mask_repr_small(v):
    repr_printer(v.S, "v.S")
    assert repr(v.S) == (
        '"v.S"             nvals  size  dtype\n'
        "StructuralMask  \n"
        "of grblas.Vector      3     5   FP64\n"
        "------------------------------------\n"
        "  0 1  2 3  4\n"
        "  1    1    1"
    )
    repr_printer(v.V, "v.V")
    assert repr(v.V) == (
        '"v.V"             nvals  size  dtype\n'
        "ValueMask       \n"
        "of grblas.Vector      3     5   FP64\n"
        "------------------------------------\n"
        "  0 1  2 3  4\n"
        "  0    1    1"
    )
    repr_printer(~v.S, "~v.S")
    assert repr(~v.S) == (
        '"~v.S"                      nvals  size  dtype\n'
        "ComplementedStructuralMask\n"
        "of grblas.Vector                3     5   FP64\n"
        "----------------------------------------------\n"
        "  0 1  2 3  4\n"
        "  0    0    0"
    )
    repr_printer(~v.V, "~v.V")
    assert repr(~v.V) == (
        '"~v.V"                 nvals  size  dtype\n'
        "ComplementedValueMask\n"
        "of grblas.Vector           3     5   FP64\n"
        "-----------------------------------------\n"
        "  0 1  2 3  4\n"
        "  1    0    0"
    )


def test_vector_mask_repr_large(w):
    with pd.option_context("display.max_columns", 26, "display.width", 100):
        repr_printer(w.S, "w.S", indent=8)
        assert repr(w.S) == (
            '"w.S"             nvals  size  dtype\n'
            "StructuralMask  \n"
            "of grblas.Vector      4    77  INT64\n"
            "------------------------------------\n"
            " 0  1  2  3  4  5  6  7  8  9  10 11 12  ... 64 65 66 67 68 69 70 71 72 73 74 75 76\n"
            "  1              1                       ...  1              1                     "
        )
        repr_printer(w.V, "w.V", indent=8)
        assert repr(w.V) == (
            '"w.V"             nvals  size  dtype\n'
            "ValueMask       \n"
            "of grblas.Vector      4    77  INT64\n"
            "------------------------------------\n"
            " 0  1  2  3  4  5  6  7  8  9  10 11 12  ... 64 65 66 67 68 69 70 71 72 73 74 75 76\n"
            "  1              1                       ...  1              1                     "
        )
        repr_printer(~w.S, "~w.S", indent=8)
        assert repr(~w.S) == (
            '"~w.S"                      nvals  size  dtype\n'
            "ComplementedStructuralMask\n"
            "of grblas.Vector                4    77  INT64\n"
            "----------------------------------------------\n"
            " 0  1  2  3  4  5  6  7  8  9  10 11 12  ... 64 65 66 67 68 69 70 71 72 73 74 75 76\n"
            "  0              0                       ...  0              0                     "
        )
        repr_printer(~w.V, "~w.V", indent=8)
        assert repr(~w.V) == (
            '"~w.V"                 nvals  size  dtype\n'
            "ComplementedValueMask\n"
            "of grblas.Vector           4    77  INT64\n"
            "-----------------------------------------\n"
            " 0  1  2  3  4  5  6  7  8  9  10 11 12  ... 64 65 66 67 68 69 70 71 72 73 74 75 76\n"
            "  0              0                       ...  0              0                     "
        )


def test_scalar_repr(s, t):
    repr_printer(s, "s")
    assert repr(s) == ('"s_1"          value  dtype\n' "grblas.Scalar     42  INT64")
    assert repr(t) == ('"t"            value  dtype\n' "grblas.Scalar   None  INT64")


def test_no_pandas_repr_html(A, C, v, w):
    # This is a bit of a hack...
    formatting.has_pandas = False
    try:
        html_printer(A, "A", indent=8)
        assert repr_html(A) == (
            '<div><details><summary style="display:list-item; outline:none;"><tt>A<sub>1</sub></tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>grblas.Matrix</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>3</td>\n"
            "    <td>1</td>\n"
            "    <td>5</td>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><em>(Install</em> <tt>pandas</tt> <em>to see a preview of the data)</em></details></div>"
        )
        html_printer(A.T, "A.T", indent=8)
        assert repr_html(A.T) == (
            '<div><details><summary style="display:list-item; outline:none;"><tt>A<sub>1</sub>.T</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>grblas.TransposedMatrix</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>3</td>\n"
            "    <td>5</td>\n"
            "    <td>1</td>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><em>(Install</em> <tt>pandas</tt> <em>to see a preview of the data)</em></details></div>"
        )
        html_printer(C.S, "C.S", indent=8)
        assert repr_html(C.S) == (
            '<div><details><summary style="display:list-item; outline:none;"><tt>C.S</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>StructuralMask\n'
            "of\n"
            "grblas.Matrix</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>8</td>\n"
            "    <td>70</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><em>(Install</em> <tt>pandas</tt> <em>to see a preview of the data)</em></details></div>"
        )
        html_printer(v, "v", indent=8)
        assert repr_html(v) == (
            '<div><details><summary style="display:list-item; outline:none;"><tt>v</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>grblas.Vector</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>size</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>3</td>\n"
            "    <td>5</td>\n"
            "    <td>FP64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><em>(Install</em> <tt>pandas</tt> <em>to see a preview of the data)</em></details></div>"
        )
        html_printer(~w.V, "~w.V", indent=8)
        assert repr_html(~w.V) == (
            '<div><details><summary style="display:list-item; outline:none;"><tt>~w.V</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>ComplementedValueMask\n'
            "of\n"
            "grblas.Vector</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>size</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>4</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><em>(Install</em> <tt>pandas</tt> <em>to see a preview of the data)</em></details></div>"
        )
    finally:
        formatting.has_pandas = True


def test_matrix_repr_html_small(A, B):
    html_printer(A, "A")
    assert repr_html(A) == (
        '<div><details open><summary style="display:list-item; outline:none;"><tt>A<sub>1</sub></tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>grblas.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "      <th>1</th>\n"
        "      <th>2</th>\n"
        "      <th>3</th>\n"
        "      <th>4</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>0</td>\n"
        "      <td></td>\n"
        "      <td>1</td>\n"
        "      <td></td>\n"
        "      <td>2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )
    html_printer(B, "B")
    assert repr_html(B) == (
        '<div><details open><summary style="display:list-item; outline:none;"><tt>B<sub>1</sub></tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>grblas.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>1</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>10</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>1</th>\n"
        "      <td></td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>2</th>\n"
        "      <td>20</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>3</th>\n"
        "      <td></td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>4</th>\n"
        "      <td>30</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )
    html_printer(B.T, "B.T")
    assert repr_html(B.T) == (
        '<div><details open><summary style="display:list-item; outline:none;"><tt>B<sub>1</sub>.T</tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>grblas.TransposedMatrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "      <th>1</th>\n"
        "      <th>2</th>\n"
        "      <th>3</th>\n"
        "      <th>4</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>10</td>\n"
        "      <td></td>\n"
        "      <td>20</td>\n"
        "      <td></td>\n"
        "      <td>30</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )


def test_matrix_mask_repr_html_small(A):
    html_printer(A.S, "A.S")
    assert repr_html(A.S) == (
        '<div><details open><summary style="display:list-item; outline:none;"><tt>A<sub>1</sub>.S</tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>StructuralMask\n'
        "of\n"
        "grblas.Matrix</pre></td>\n"
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "      <th>1</th>\n"
        "      <th>2</th>\n"
        "      <th>3</th>\n"
        "      <th>4</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>1</td>\n"
        "      <td></td>\n"
        "      <td>1</td>\n"
        "      <td></td>\n"
        "      <td>1</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )
    html_printer(A.V, "A.V")
    assert repr_html(A.V) == (
        '<div><details open><summary style="display:list-item; outline:none;"><tt>A<sub>1</sub>.V</tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>ValueMask\n'
        "of\n"
        "grblas.Matrix</pre></td>\n"
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "      <th>1</th>\n"
        "      <th>2</th>\n"
        "      <th>3</th>\n"
        "      <th>4</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>0</td>\n"
        "      <td></td>\n"
        "      <td>1</td>\n"
        "      <td></td>\n"
        "      <td>1</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )
    html_printer(~A.S, "~A.S")
    assert repr_html(~A.S) == (
        '<div><details open><summary style="display:list-item; outline:none;"><tt>~A<sub>1</sub>.S</tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>ComplementedStructuralMask\n'
        "of\n"
        "grblas.Matrix</pre></td>\n"
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "      <th>1</th>\n"
        "      <th>2</th>\n"
        "      <th>3</th>\n"
        "      <th>4</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>0</td>\n"
        "      <td></td>\n"
        "      <td>0</td>\n"
        "      <td></td>\n"
        "      <td>0</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )
    html_printer(~A.V, "~A.V")
    assert repr_html(~A.V) == (
        '<div><details open><summary style="display:list-item; outline:none;"><tt>~A<sub>1</sub>.V</tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>ComplementedValueMask\n'
        "of\n"
        "grblas.Matrix</pre></td>\n"
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "      <th>1</th>\n"
        "      <th>2</th>\n"
        "      <th>3</th>\n"
        "      <th>4</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>1</td>\n"
        "      <td></td>\n"
        "      <td>0</td>\n"
        "      <td></td>\n"
        "      <td>0</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )


def test_matrix_repr_html_large(C, D):
    with pd.option_context("display.max_columns", 20):
        html_printer(C, "C", indent=8)
        assert repr_html(C) == (
            '<div><details open><summary style="display:list-item; outline:none;"><tt>C</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>grblas.Matrix</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>8</td>\n"
            "    <td>70</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><div>\n"
            "<style scoped>\n"
            "    .dataframe tbody tr th:only-of-type {\n"
            "        vertical-align: middle;\n"
            "    }\n"
            "\n"
            "    .dataframe tbody tr th {\n"
            "        vertical-align: top;\n"
            "    }\n"
            "\n"
            "    .dataframe thead th {\n"
            "        text-align: right;\n"
            "    }\n"
            "</style>\n"
            '<table border="1" class="dataframe">\n'
            "  <thead>\n"
            '    <tr style="text-align: right;">\n'
            "      <th></th>\n"
            "      <th>0</th>\n"
            "      <th>1</th>\n"
            "      <th>2</th>\n"
            "      <th>3</th>\n"
            "      <th>4</th>\n"
            "      <th>5</th>\n"
            "      <th>6</th>\n"
            "      <th>7</th>\n"
            "      <th>8</th>\n"
            "      <th>9</th>\n"
            "      <th>...</th>\n"
            "      <th>67</th>\n"
            "      <th>68</th>\n"
            "      <th>69</th>\n"
            "      <th>70</th>\n"
            "      <th>71</th>\n"
            "      <th>72</th>\n"
            "      <th>73</th>\n"
            "      <th>74</th>\n"
            "      <th>75</th>\n"
            "      <th>76</th>\n"
            "    </tr>\n"
            "  </thead>\n"
            "  <tbody>\n"
            "    <tr>\n"
            "      <th>0</th>\n"
            + "      <td></td>\n" * 4
            + "      <td>0</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>5</td>\n"
            + "      <td></td>\n" * 4
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>1</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>2</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>3</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>4</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>...</th>\n" + "      <td>...</td>\n" * 21 + "    </tr>\n"
            "    <tr>\n"
            "      <th>65</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>66</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>67</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>68</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>69</th>\n"
            + "      <td></td>\n" * 4
            + "      <td>4</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>8</td>\n"
            + "      <td></td>\n" * 4
            + "    </tr>\n"
            "  </tbody>\n"
            "</table>\n"
            "</div></details></div>"
        )
        html_printer(C.T, "C.T", indent=8)
        assert repr_html(C.T) == (
            '<div><details open><summary style="display:list-item; outline:none;"><tt>C.T</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>grblas.TransposedMatrix</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>8</td>\n"
            "    <td>77</td>\n"
            "    <td>70</td>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><div>\n"
            "<style scoped>\n"
            "    .dataframe tbody tr th:only-of-type {\n"
            "        vertical-align: middle;\n"
            "    }\n"
            "\n"
            "    .dataframe tbody tr th {\n"
            "        vertical-align: top;\n"
            "    }\n"
            "\n"
            "    .dataframe thead th {\n"
            "        text-align: right;\n"
            "    }\n"
            "</style>\n"
            '<table border="1" class="dataframe">\n'
            "  <thead>\n"
            '    <tr style="text-align: right;">\n'
            "      <th></th>\n"
            "      <th>0</th>\n"
            "      <th>1</th>\n"
            "      <th>2</th>\n"
            "      <th>3</th>\n"
            "      <th>4</th>\n"
            "      <th>5</th>\n"
            "      <th>6</th>\n"
            "      <th>7</th>\n"
            "      <th>8</th>\n"
            "      <th>9</th>\n"
            "      <th>...</th>\n"
            "      <th>60</th>\n"
            "      <th>61</th>\n"
            "      <th>62</th>\n"
            "      <th>63</th>\n"
            "      <th>64</th>\n"
            "      <th>65</th>\n"
            "      <th>66</th>\n"
            "      <th>67</th>\n"
            "      <th>68</th>\n"
            "      <th>69</th>\n"
            "    </tr>\n"
            "  </thead>\n"
            "  <tbody>\n"
            "    <tr>\n"
            "      <th>0</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>1</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>2</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>3</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>4</th>\n"
            "      <td>0</td>\n" + "      <td></td>\n" * 8 + "      <td>2</td>\n"
            "      <td>...</td>\n"
            "      <td>3</td>\n" + "      <td></td>\n" * 8 + "      <td>4</td>\n"
            "    </tr>\n"
            "    <tr>\n"
            "      <th>...</th>\n" + "      <td>...</td>\n" * 21 + "    </tr>\n"
            "    <tr>\n"
            "      <th>72</th>\n"
            "      <td>5</td>\n" + "      <td></td>\n" * 8 + "      <td>6</td>\n"
            "      <td>...</td>\n"
            "      <td>7</td>\n" + "      <td></td>\n" * 8 + "      <td>8</td>\n"
            "    </tr>\n"
            "    <tr>\n"
            "      <th>73</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>74</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>75</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>76</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "  </tbody>\n"
            "</table>\n"
            "</div></details></div>"
        )
        html_printer(D, "D", indent=8)
        assert repr_html(D) == (
            '<div><details open><summary style="display:list-item; outline:none;"><tt>D<sub>skinny_in_one_dim</sub></tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>grblas.Matrix</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>4</td>\n"
            "    <td>70</td>\n"
            "    <td>5</td>\n"
            "    <td>BOOL</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><div>\n"
            "<style scoped>\n"
            "    .dataframe tbody tr th:only-of-type {\n"
            "        vertical-align: middle;\n"
            "    }\n"
            "\n"
            "    .dataframe tbody tr th {\n"
            "        vertical-align: top;\n"
            "    }\n"
            "\n"
            "    .dataframe thead th {\n"
            "        text-align: right;\n"
            "    }\n"
            "</style>\n"
            '<table border="1" class="dataframe">\n'
            "  <thead>\n"
            '    <tr style="text-align: right;">\n'
            "      <th></th>\n"
            "      <th>0</th>\n"
            "      <th>1</th>\n"
            "      <th>2</th>\n"
            "      <th>3</th>\n"
            "      <th>4</th>\n"
            "    </tr>\n"
            "  </thead>\n"
            "  <tbody>\n"
            "    <tr>\n"
            "      <th>0</th>\n" + "      <td></td>\n" * 4 + "      <td>True</td>\n"
            "    </tr>\n"
            "    <tr>\n"
            "      <th>1</th>\n" + "      <td></td>\n" * 5 + "    </tr>\n"
            "    <tr>\n"
            "      <th>2</th>\n" + "      <td></td>\n" * 5 + "    </tr>\n"
            "    <tr>\n"
            "      <th>3</th>\n" + "      <td></td>\n" * 5 + "    </tr>\n"
            "    <tr>\n"
            "      <th>4</th>\n" + "      <td></td>\n" * 5 + "    </tr>\n"
            "    <tr>\n"
            "      <th>...</th>\n" + "      <td>...</td>\n" * 5 + "    </tr>\n"
            "    <tr>\n"
            "      <th>65</th>\n" + "      <td></td>\n" * 5 + "    </tr>\n"
            "    <tr>\n"
            "      <th>66</th>\n" + "      <td></td>\n" * 5 + "    </tr>\n"
            "    <tr>\n"
            "      <th>67</th>\n" + "      <td></td>\n" * 5 + "    </tr>\n"
            "    <tr>\n"
            "      <th>68</th>\n" + "      <td></td>\n" * 5 + "    </tr>\n"
            "    <tr>\n"
            "      <th>69</th>\n" + "      <td></td>\n" * 4 + "      <td>False</td>\n"
            "    </tr>\n"
            "  </tbody>\n"
            "</table>\n"
            "</div></details></div>"
        )
        html_printer(D.T, "D.T", indent=8)
        assert repr_html(D.T) == (
            '<div><details open><summary style="display:list-item; outline:none;"><tt>D<sub>skinny_in_one_dim</sub>.T</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>grblas.TransposedMatrix</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>4</td>\n"
            "    <td>5</td>\n"
            "    <td>70</td>\n"
            "    <td>BOOL</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><div>\n"
            "<style scoped>\n"
            "    .dataframe tbody tr th:only-of-type {\n"
            "        vertical-align: middle;\n"
            "    }\n"
            "\n"
            "    .dataframe tbody tr th {\n"
            "        vertical-align: top;\n"
            "    }\n"
            "\n"
            "    .dataframe thead th {\n"
            "        text-align: right;\n"
            "    }\n"
            "</style>\n"
            '<table border="1" class="dataframe">\n'
            "  <thead>\n"
            '    <tr style="text-align: right;">\n'
            "      <th></th>\n"
            "      <th>0</th>\n"
            "      <th>1</th>\n"
            "      <th>2</th>\n"
            "      <th>3</th>\n"
            "      <th>4</th>\n"
            "      <th>5</th>\n"
            "      <th>6</th>\n"
            "      <th>7</th>\n"
            "      <th>8</th>\n"
            "      <th>9</th>\n"
            "      <th>...</th>\n"
            "      <th>60</th>\n"
            "      <th>61</th>\n"
            "      <th>62</th>\n"
            "      <th>63</th>\n"
            "      <th>64</th>\n"
            "      <th>65</th>\n"
            "      <th>66</th>\n"
            "      <th>67</th>\n"
            "      <th>68</th>\n"
            "      <th>69</th>\n"
            "    </tr>\n"
            "  </thead>\n"
            "  <tbody>\n"
            "    <tr>\n"
            "      <th>0</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>1</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>2</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>3</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>4</th>\n"
            "      <td>True</td>\n" + "      <td></td>\n" * 8 + "      <td>False</td>\n"
            "      <td>...</td>\n"
            "      <td>True</td>\n" + "      <td></td>\n" * 8 + "      <td>False</td>\n"
            "    </tr>\n"
            "  </tbody>\n"
            "</table>\n"
            "</div></details></div>"
        )


def test_matrix_mask_repr_html_large(C):
    with pd.option_context("display.max_columns", 20):
        html_printer(C.S, "C.S", indent=8)
        assert repr_html(C.S) == (
            '<div><details open><summary style="display:list-item; outline:none;"><tt>C.S</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>StructuralMask\n'
            "of\n"
            "grblas.Matrix</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>8</td>\n"
            "    <td>70</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><div>\n"
            "<style scoped>\n"
            "    .dataframe tbody tr th:only-of-type {\n"
            "        vertical-align: middle;\n"
            "    }\n"
            "\n"
            "    .dataframe tbody tr th {\n"
            "        vertical-align: top;\n"
            "    }\n"
            "\n"
            "    .dataframe thead th {\n"
            "        text-align: right;\n"
            "    }\n"
            "</style>\n"
            '<table border="1" class="dataframe">\n'
            "  <thead>\n"
            '    <tr style="text-align: right;">\n'
            "      <th></th>\n"
            "      <th>0</th>\n"
            "      <th>1</th>\n"
            "      <th>2</th>\n"
            "      <th>3</th>\n"
            "      <th>4</th>\n"
            "      <th>5</th>\n"
            "      <th>6</th>\n"
            "      <th>7</th>\n"
            "      <th>8</th>\n"
            "      <th>9</th>\n"
            "      <th>...</th>\n"
            "      <th>67</th>\n"
            "      <th>68</th>\n"
            "      <th>69</th>\n"
            "      <th>70</th>\n"
            "      <th>71</th>\n"
            "      <th>72</th>\n"
            "      <th>73</th>\n"
            "      <th>74</th>\n"
            "      <th>75</th>\n"
            "      <th>76</th>\n"
            "    </tr>\n"
            "  </thead>\n"
            "  <tbody>\n"
            "    <tr>\n"
            "      <th>0</th>\n"
            + "      <td></td>\n" * 4
            + "      <td>1</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>1</td>\n"
            + "      <td></td>\n" * 4
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>1</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>2</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>3</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>4</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>...</th>\n" + "      <td>...</td>\n" * 21 + "    </tr>\n"
            "    <tr>\n"
            "      <th>65</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>66</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>67</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>68</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>69</th>\n"
            + "      <td></td>\n" * 4
            + "      <td>1</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>1</td>\n"
            + "      <td></td>\n" * 4
            + "    </tr>\n"
            "  </tbody>\n"
            "</table>\n"
            "</div></details></div>"
        )
        html_printer(C.V, "C.V", indent=8)
        assert repr_html(C.V) == (
            '<div><details open><summary style="display:list-item; outline:none;"><tt>C.V</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>ValueMask\n'
            "of\n"
            "grblas.Matrix</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>8</td>\n"
            "    <td>70</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><div>\n"
            "<style scoped>\n"
            "    .dataframe tbody tr th:only-of-type {\n"
            "        vertical-align: middle;\n"
            "    }\n"
            "\n"
            "    .dataframe tbody tr th {\n"
            "        vertical-align: top;\n"
            "    }\n"
            "\n"
            "    .dataframe thead th {\n"
            "        text-align: right;\n"
            "    }\n"
            "</style>\n"
            '<table border="1" class="dataframe">\n'
            "  <thead>\n"
            '    <tr style="text-align: right;">\n'
            "      <th></th>\n"
            "      <th>0</th>\n"
            "      <th>1</th>\n"
            "      <th>2</th>\n"
            "      <th>3</th>\n"
            "      <th>4</th>\n"
            "      <th>5</th>\n"
            "      <th>6</th>\n"
            "      <th>7</th>\n"
            "      <th>8</th>\n"
            "      <th>9</th>\n"
            "      <th>...</th>\n"
            "      <th>67</th>\n"
            "      <th>68</th>\n"
            "      <th>69</th>\n"
            "      <th>70</th>\n"
            "      <th>71</th>\n"
            "      <th>72</th>\n"
            "      <th>73</th>\n"
            "      <th>74</th>\n"
            "      <th>75</th>\n"
            "      <th>76</th>\n"
            "    </tr>\n"
            "  </thead>\n"
            "  <tbody>\n"
            "    <tr>\n"
            "      <th>0</th>\n"
            + "      <td></td>\n" * 4
            + "      <td>0</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>1</td>\n"
            + "      <td></td>\n" * 4
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>1</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>2</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>3</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>4</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>...</th>\n" + "      <td>...</td>\n" * 21 + "    </tr>\n"
            "    <tr>\n"
            "      <th>65</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>66</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>67</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>68</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>69</th>\n"
            + "      <td></td>\n" * 4
            + "      <td>1</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>1</td>\n"
            + "      <td></td>\n" * 4
            + "    </tr>\n"
            "  </tbody>\n"
            "</table>\n"
            "</div></details></div>"
        )
        html_printer(~C.S, "~C.S", indent=8)
        assert repr_html(~C.S) == (
            '<div><details open><summary style="display:list-item; outline:none;"><tt>~C.S</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>ComplementedStructuralMask\n'
            "of\n"
            "grblas.Matrix</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>8</td>\n"
            "    <td>70</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><div>\n"
            "<style scoped>\n"
            "    .dataframe tbody tr th:only-of-type {\n"
            "        vertical-align: middle;\n"
            "    }\n"
            "\n"
            "    .dataframe tbody tr th {\n"
            "        vertical-align: top;\n"
            "    }\n"
            "\n"
            "    .dataframe thead th {\n"
            "        text-align: right;\n"
            "    }\n"
            "</style>\n"
            '<table border="1" class="dataframe">\n'
            "  <thead>\n"
            '    <tr style="text-align: right;">\n'
            "      <th></th>\n"
            "      <th>0</th>\n"
            "      <th>1</th>\n"
            "      <th>2</th>\n"
            "      <th>3</th>\n"
            "      <th>4</th>\n"
            "      <th>5</th>\n"
            "      <th>6</th>\n"
            "      <th>7</th>\n"
            "      <th>8</th>\n"
            "      <th>9</th>\n"
            "      <th>...</th>\n"
            "      <th>67</th>\n"
            "      <th>68</th>\n"
            "      <th>69</th>\n"
            "      <th>70</th>\n"
            "      <th>71</th>\n"
            "      <th>72</th>\n"
            "      <th>73</th>\n"
            "      <th>74</th>\n"
            "      <th>75</th>\n"
            "      <th>76</th>\n"
            "    </tr>\n"
            "  </thead>\n"
            "  <tbody>\n"
            "    <tr>\n"
            "      <th>0</th>\n"
            + "      <td></td>\n" * 4
            + "      <td>0</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>0</td>\n"
            + "      <td></td>\n" * 4
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>1</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>2</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>3</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>4</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>...</th>\n" + "      <td>...</td>\n" * 21 + "    </tr>\n"
            "    <tr>\n"
            "      <th>65</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>66</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>67</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>68</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>69</th>\n"
            + "      <td></td>\n" * 4
            + "      <td>0</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>0</td>\n"
            + "      <td></td>\n" * 4
            + "    </tr>\n"
            "  </tbody>\n"
            "</table>\n"
            "</div></details></div>"
        )
        html_printer(~C.V, "~C.V", indent=8)
        assert repr_html(~C.V) == (
            '<div><details open><summary style="display:list-item; outline:none;"><tt>~C.V</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>ComplementedValueMask\n'
            "of\n"
            "grblas.Matrix</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>8</td>\n"
            "    <td>70</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><div>\n"
            "<style scoped>\n"
            "    .dataframe tbody tr th:only-of-type {\n"
            "        vertical-align: middle;\n"
            "    }\n"
            "\n"
            "    .dataframe tbody tr th {\n"
            "        vertical-align: top;\n"
            "    }\n"
            "\n"
            "    .dataframe thead th {\n"
            "        text-align: right;\n"
            "    }\n"
            "</style>\n"
            '<table border="1" class="dataframe">\n'
            "  <thead>\n"
            '    <tr style="text-align: right;">\n'
            "      <th></th>\n"
            "      <th>0</th>\n"
            "      <th>1</th>\n"
            "      <th>2</th>\n"
            "      <th>3</th>\n"
            "      <th>4</th>\n"
            "      <th>5</th>\n"
            "      <th>6</th>\n"
            "      <th>7</th>\n"
            "      <th>8</th>\n"
            "      <th>9</th>\n"
            "      <th>...</th>\n"
            "      <th>67</th>\n"
            "      <th>68</th>\n"
            "      <th>69</th>\n"
            "      <th>70</th>\n"
            "      <th>71</th>\n"
            "      <th>72</th>\n"
            "      <th>73</th>\n"
            "      <th>74</th>\n"
            "      <th>75</th>\n"
            "      <th>76</th>\n"
            "    </tr>\n"
            "  </thead>\n"
            "  <tbody>\n"
            "    <tr>\n"
            "      <th>0</th>\n"
            + "      <td></td>\n" * 4
            + "      <td>1</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>0</td>\n"
            + "      <td></td>\n" * 4
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>1</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>2</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>3</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>4</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>...</th>\n" + "      <td>...</td>\n" * 21 + "    </tr>\n"
            "    <tr>\n"
            "      <th>65</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>66</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>67</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>68</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>69</th>\n"
            + "      <td></td>\n" * 4
            + "      <td>0</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>0</td>\n"
            + "      <td></td>\n" * 4
            + "    </tr>\n"
            "  </tbody>\n"
            "</table>\n"
            "</div></details></div>"
        )


def test_vector_repr_html_small(v):
    html_printer(v, "v")
    assert repr_html(v) == (
        '<div><details open><summary style="display:list-item; outline:none;"><tt>v</tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>grblas.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "      <th>1</th>\n"
        "      <th>2</th>\n"
        "      <th>3</th>\n"
        "      <th>4</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        "      <td>0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )


def test_vector_repr_html_large(w):
    with pd.option_context("display.max_columns", 20):
        html_printer(w, "w", indent=8)
        assert repr_html(w) == (
            '<div><details open><summary style="display:list-item; outline:none;"><tt>w</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>grblas.Vector</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>size</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>4</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><div>\n"
            "<style scoped>\n"
            "    .dataframe tbody tr th:only-of-type {\n"
            "        vertical-align: middle;\n"
            "    }\n"
            "\n"
            "    .dataframe tbody tr th {\n"
            "        vertical-align: top;\n"
            "    }\n"
            "\n"
            "    .dataframe thead th {\n"
            "        text-align: right;\n"
            "    }\n"
            "</style>\n"
            '<table border="1" class="dataframe">\n'
            "  <thead>\n"
            '    <tr style="text-align: right;">\n'
            "      <th></th>\n"
            "      <th>0</th>\n"
            "      <th>1</th>\n"
            "      <th>2</th>\n"
            "      <th>3</th>\n"
            "      <th>4</th>\n"
            "      <th>5</th>\n"
            "      <th>6</th>\n"
            "      <th>7</th>\n"
            "      <th>8</th>\n"
            "      <th>9</th>\n"
            "      <th>...</th>\n"
            "      <th>67</th>\n"
            "      <th>68</th>\n"
            "      <th>69</th>\n"
            "      <th>70</th>\n"
            "      <th>71</th>\n"
            "      <th>72</th>\n"
            "      <th>73</th>\n"
            "      <th>74</th>\n"
            "      <th>75</th>\n"
            "      <th>76</th>\n"
            "    </tr>\n"
            "  </thead>\n"
            "  <tbody>\n"
            "    <tr>\n"
            "      <th></th>\n"
            "      <td>1</td>\n"
            + "      <td></td>\n" * 4
            + "      <td>2</td>\n"
            + "      <td></td>\n" * 4
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 2
            + "      <td>4</td>\n"
            + "      <td></td>\n" * 7
            + "    </tr>\n"
            "  </tbody>\n"
            "</table>\n"
            "</div></details></div>"
        )


def test_vector_mask_repr_html_small(v):
    html_printer(v.S, "v.S")
    assert repr_html(v.S) == (
        '<div><details open><summary style="display:list-item; outline:none;"><tt>v.S</tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>StructuralMask\n'
        "of\n"
        "grblas.Vector</pre></td>\n"
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "      <th>1</th>\n"
        "      <th>2</th>\n"
        "      <th>3</th>\n"
        "      <th>4</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        "      <td>1</td>\n"
        "      <td></td>\n"
        "      <td>1</td>\n"
        "      <td></td>\n"
        "      <td>1</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )
    html_printer(v.V, "v.V")
    assert repr_html(v.V) == (
        '<div><details open><summary style="display:list-item; outline:none;"><tt>v.V</tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>ValueMask\n'
        "of\n"
        "grblas.Vector</pre></td>\n"
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "      <th>1</th>\n"
        "      <th>2</th>\n"
        "      <th>3</th>\n"
        "      <th>4</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        "      <td>0</td>\n"
        "      <td></td>\n"
        "      <td>1</td>\n"
        "      <td></td>\n"
        "      <td>1</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )
    html_printer(~v.S, "~v.S")
    assert repr_html(~v.S) == (
        '<div><details open><summary style="display:list-item; outline:none;"><tt>~v.S</tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>ComplementedStructuralMask\n'
        "of\n"
        "grblas.Vector</pre></td>\n"
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "      <th>1</th>\n"
        "      <th>2</th>\n"
        "      <th>3</th>\n"
        "      <th>4</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        "      <td>0</td>\n"
        "      <td></td>\n"
        "      <td>0</td>\n"
        "      <td></td>\n"
        "      <td>0</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )
    html_printer(~v.V, "~v.V")
    assert repr_html(~v.V) == (
        '<div><details open><summary style="display:list-item; outline:none;"><tt>~v.V</tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>ComplementedValueMask\n'
        "of\n"
        "grblas.Vector</pre></td>\n"
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "      <th>1</th>\n"
        "      <th>2</th>\n"
        "      <th>3</th>\n"
        "      <th>4</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        "      <td>1</td>\n"
        "      <td></td>\n"
        "      <td>0</td>\n"
        "      <td></td>\n"
        "      <td>0</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )


def test_vector_mask_repr_html_large(w):
    with pd.option_context("display.max_columns", 20):
        html_printer(w.S, "w.S", indent=8)
        assert repr_html(w.S) == (
            '<div><details open><summary style="display:list-item; outline:none;"><tt>w.S</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>StructuralMask\n'
            "of\n"
            "grblas.Vector</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>size</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>4</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><div>\n"
            "<style scoped>\n"
            "    .dataframe tbody tr th:only-of-type {\n"
            "        vertical-align: middle;\n"
            "    }\n"
            "\n"
            "    .dataframe tbody tr th {\n"
            "        vertical-align: top;\n"
            "    }\n"
            "\n"
            "    .dataframe thead th {\n"
            "        text-align: right;\n"
            "    }\n"
            "</style>\n"
            '<table border="1" class="dataframe">\n'
            "  <thead>\n"
            '    <tr style="text-align: right;">\n'
            "      <th></th>\n"
            "      <th>0</th>\n"
            "      <th>1</th>\n"
            "      <th>2</th>\n"
            "      <th>3</th>\n"
            "      <th>4</th>\n"
            "      <th>5</th>\n"
            "      <th>6</th>\n"
            "      <th>7</th>\n"
            "      <th>8</th>\n"
            "      <th>9</th>\n"
            "      <th>...</th>\n"
            "      <th>67</th>\n"
            "      <th>68</th>\n"
            "      <th>69</th>\n"
            "      <th>70</th>\n"
            "      <th>71</th>\n"
            "      <th>72</th>\n"
            "      <th>73</th>\n"
            "      <th>74</th>\n"
            "      <th>75</th>\n"
            "      <th>76</th>\n"
            "    </tr>\n"
            "  </thead>\n"
            "  <tbody>\n"
            "    <tr>\n"
            "      <th></th>\n"
            "      <td>1</td>\n"
            + "      <td></td>\n" * 4
            + "      <td>1</td>\n"
            + "      <td></td>\n" * 4
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 2
            + "      <td>1</td>\n"
            + "      <td></td>\n" * 7
            + "    </tr>\n"
            "  </tbody>\n"
            "</table>\n"
            "</div></details></div>"
        )
        html_printer(w.V, "w.V", indent=8)
        assert repr_html(w.V) == (
            '<div><details open><summary style="display:list-item; outline:none;"><tt>w.V</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>ValueMask\n'
            "of\n"
            "grblas.Vector</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>size</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>4</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><div>\n"
            "<style scoped>\n"
            "    .dataframe tbody tr th:only-of-type {\n"
            "        vertical-align: middle;\n"
            "    }\n"
            "\n"
            "    .dataframe tbody tr th {\n"
            "        vertical-align: top;\n"
            "    }\n"
            "\n"
            "    .dataframe thead th {\n"
            "        text-align: right;\n"
            "    }\n"
            "</style>\n"
            '<table border="1" class="dataframe">\n'
            "  <thead>\n"
            '    <tr style="text-align: right;">\n'
            "      <th></th>\n"
            "      <th>0</th>\n"
            "      <th>1</th>\n"
            "      <th>2</th>\n"
            "      <th>3</th>\n"
            "      <th>4</th>\n"
            "      <th>5</th>\n"
            "      <th>6</th>\n"
            "      <th>7</th>\n"
            "      <th>8</th>\n"
            "      <th>9</th>\n"
            "      <th>...</th>\n"
            "      <th>67</th>\n"
            "      <th>68</th>\n"
            "      <th>69</th>\n"
            "      <th>70</th>\n"
            "      <th>71</th>\n"
            "      <th>72</th>\n"
            "      <th>73</th>\n"
            "      <th>74</th>\n"
            "      <th>75</th>\n"
            "      <th>76</th>\n"
            "    </tr>\n"
            "  </thead>\n"
            "  <tbody>\n"
            "    <tr>\n"
            "      <th></th>\n"
            "      <td>1</td>\n"
            + "      <td></td>\n" * 4
            + "      <td>1</td>\n"
            + "      <td></td>\n" * 4
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 2
            + "      <td>1</td>\n"
            + "      <td></td>\n" * 7
            + "    </tr>\n"
            "  </tbody>\n"
            "</table>\n"
            "</div></details></div>"
        )
        html_printer(~w.S, "~w.S", indent=8)
        assert repr_html(~w.S) == (
            '<div><details open><summary style="display:list-item; outline:none;"><tt>~w.S</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>ComplementedStructuralMask\n'
            "of\n"
            "grblas.Vector</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>size</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>4</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><div>\n"
            "<style scoped>\n"
            "    .dataframe tbody tr th:only-of-type {\n"
            "        vertical-align: middle;\n"
            "    }\n"
            "\n"
            "    .dataframe tbody tr th {\n"
            "        vertical-align: top;\n"
            "    }\n"
            "\n"
            "    .dataframe thead th {\n"
            "        text-align: right;\n"
            "    }\n"
            "</style>\n"
            '<table border="1" class="dataframe">\n'
            "  <thead>\n"
            '    <tr style="text-align: right;">\n'
            "      <th></th>\n"
            "      <th>0</th>\n"
            "      <th>1</th>\n"
            "      <th>2</th>\n"
            "      <th>3</th>\n"
            "      <th>4</th>\n"
            "      <th>5</th>\n"
            "      <th>6</th>\n"
            "      <th>7</th>\n"
            "      <th>8</th>\n"
            "      <th>9</th>\n"
            "      <th>...</th>\n"
            "      <th>67</th>\n"
            "      <th>68</th>\n"
            "      <th>69</th>\n"
            "      <th>70</th>\n"
            "      <th>71</th>\n"
            "      <th>72</th>\n"
            "      <th>73</th>\n"
            "      <th>74</th>\n"
            "      <th>75</th>\n"
            "      <th>76</th>\n"
            "    </tr>\n"
            "  </thead>\n"
            "  <tbody>\n"
            "    <tr>\n"
            "      <th></th>\n"
            "      <td>0</td>\n"
            + "      <td></td>\n" * 4
            + "      <td>0</td>\n"
            + "      <td></td>\n" * 4
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 2
            + "      <td>0</td>\n"
            + "      <td></td>\n" * 7
            + "    </tr>\n"
            "  </tbody>\n"
            "</table>\n"
            "</div></details></div>"
        )
        html_printer(~w.V, "~w.V", indent=8)
        assert repr_html(~w.V) == (
            '<div><details open><summary style="display:list-item; outline:none;"><tt>~w.V</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>ComplementedValueMask\n'
            "of\n"
            "grblas.Vector</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>size</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>4</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><div>\n"
            "<style scoped>\n"
            "    .dataframe tbody tr th:only-of-type {\n"
            "        vertical-align: middle;\n"
            "    }\n"
            "\n"
            "    .dataframe tbody tr th {\n"
            "        vertical-align: top;\n"
            "    }\n"
            "\n"
            "    .dataframe thead th {\n"
            "        text-align: right;\n"
            "    }\n"
            "</style>\n"
            '<table border="1" class="dataframe">\n'
            "  <thead>\n"
            '    <tr style="text-align: right;">\n'
            "      <th></th>\n"
            "      <th>0</th>\n"
            "      <th>1</th>\n"
            "      <th>2</th>\n"
            "      <th>3</th>\n"
            "      <th>4</th>\n"
            "      <th>5</th>\n"
            "      <th>6</th>\n"
            "      <th>7</th>\n"
            "      <th>8</th>\n"
            "      <th>9</th>\n"
            "      <th>...</th>\n"
            "      <th>67</th>\n"
            "      <th>68</th>\n"
            "      <th>69</th>\n"
            "      <th>70</th>\n"
            "      <th>71</th>\n"
            "      <th>72</th>\n"
            "      <th>73</th>\n"
            "      <th>74</th>\n"
            "      <th>75</th>\n"
            "      <th>76</th>\n"
            "    </tr>\n"
            "  </thead>\n"
            "  <tbody>\n"
            "    <tr>\n"
            "      <th></th>\n"
            "      <td>0</td>\n"
            + "      <td></td>\n" * 4
            + "      <td>0</td>\n"
            + "      <td></td>\n" * 4
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 2
            + "      <td>0</td>\n"
            + "      <td></td>\n" * 7
            + "    </tr>\n"
            "  </tbody>\n"
            "</table>\n"
            "</div></details></div>"
        )


def test_scalar_repr_html(s, t):
    html_printer(s, "s")
    assert repr_html(s) == (
        "<div><tt>s<sub>1</sub></tt><div>\n"
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>grblas.Scalar</pre></td>\n'
        "    <td><pre>value</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>42</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</div>"
    )
    html_printer(t, "t")
    assert repr_html(t) == (
        "<div><tt>t</tt><div>\n"
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>grblas.Scalar</pre></td>\n'
        "    <td><pre>value</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>None</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</div>"
    )


def test_apply_repr(v):
    repr_printer(v.apply(unary.one), "v.apply(unary.one)")
    assert repr(v.apply(unary.one)) == (
        "grblas.VectorExpression   size  dtype\n"
        "v.apply(unary.one[FP64])     5   FP64\n"
        "\n"
        "Do expr.new() or other << expr to calculate the expression."
    )


def test_apply_repr_html(v):
    html_printer(v.apply(unary.one), "v.apply(unary.one)")
    assert repr_html(v.apply(unary.one)) == (
        '<div style="padding:4px;"><details><summary style="display:list-item; outline:none;"><b><tt>grblas.VectorExpression:</tt></b><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>v.apply(unary.one[FP64])</pre></td>\n'
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote><div><details open><summary style="display:list-item; outline:none;"><tt>v</tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>grblas.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "      <th>1</th>\n"
        "      <th>2</th>\n"
        "      <th>3</th>\n"
        "      <th>4</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        "      <td>0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>Do <code>expr.new()</code> or <code>other << expr</code> to calculate the expression.</em></div>"
    )


def test_mxm_repr(A, B):
    repr_printer(A.mxm(B), "A.mxm(B)")
    assert repr(A.mxm(B)) == (
        "grblas.MatrixExpression                      nrows  ncols  dtype\n"
        "A_1.mxm(B_1, op=semiring.plus_times[INT64])      1      1  INT64\n"
        "\n"
        "Do expr.new() or other << expr to calculate the expression."
    )


def test_mxm_repr_html(A, B):
    html_printer(A.mxm(B), "A.mxm(B)")
    assert repr_html(A.mxm(B)) == (
        '<div style="padding:4px;"><details><summary style="display:list-item; outline:none;"><b><tt>grblas.MatrixExpression:</tt></b><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>A<sub>1</sub>.mxm(B<sub>1</sub>, op=semiring.plus_times[INT64])</pre></td>\n'
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n" + "    <td>1</td>\n" * 2 + "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote><div><details open><summary style="display:list-item; outline:none;"><tt>A<sub>1</sub></tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>grblas.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "      <th>1</th>\n"
        "      <th>2</th>\n"
        "      <th>3</th>\n"
        "      <th>4</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>0</td>\n"
        "      <td></td>\n"
        "      <td>1</td>\n"
        "      <td></td>\n"
        "      <td>2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        '</div></details></div><div><details open><summary style="display:list-item; outline:none;"><tt>B<sub>1</sub></tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>grblas.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>1</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>10</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>1</th>\n"
        "      <td></td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>2</th>\n"
        "      <td>20</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>3</th>\n"
        "      <td></td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>4</th>\n"
        "      <td>30</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>Do <code>expr.new()</code> or <code>other << expr</code> to calculate the expression.</em></div>"
    )


def test_mxv_repr(A, v):
    repr_printer(A.mxv(v), "A.mxv(v)")
    assert repr(A.mxv(v)) == (
        "grblas.VectorExpression                   size  dtype\n"
        "A_1.mxv(v, op=semiring.plus_times[FP64])     1   FP64\n"
        "\n"
        "Do expr.new() or other << expr to calculate the expression."
    )


def test_mxv_repr_html(A, v):
    html_printer(A.mxv(v), "A.mxv(v)")
    assert repr_html(A.mxv(v)) == (
        '<div style="padding:4px;"><details><summary style="display:list-item; outline:none;"><b><tt>grblas.VectorExpression:</tt></b><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>A<sub>1</sub>.mxv(v, op=semiring.plus_times[FP64])</pre></td>\n'
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>1</td>\n"
        "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote><div><details open><summary style="display:list-item; outline:none;"><tt>A<sub>1</sub></tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>grblas.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "      <th>1</th>\n"
        "      <th>2</th>\n"
        "      <th>3</th>\n"
        "      <th>4</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>0</td>\n"
        "      <td></td>\n"
        "      <td>1</td>\n"
        "      <td></td>\n"
        "      <td>2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        '</div></details></div><div><details open><summary style="display:list-item; outline:none;"><tt>v</tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>grblas.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "      <th>1</th>\n"
        "      <th>2</th>\n"
        "      <th>3</th>\n"
        "      <th>4</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        "      <td>0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>Do <code>expr.new()</code> or <code>other << expr</code> to calculate the expression.</em></div>"
    )


def test_matrix_reduce_columns_repr_html(A):
    # This is implmeneted using the transpose of A, so make sure we're oriented correctly!
    html_printer(A.reduce_columns(), "A.reduce_columns()")
    assert repr_html(A.reduce_columns()) == (
        '<div style="padding:4px;"><details><summary style="display:list-item; outline:none;"><b><tt>grblas.VectorExpression:</tt></b><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>A<sub>1</sub>.reduce_columns(monoid.plus[INT64])</pre></td>\n'
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote><div><details open><summary style="display:list-item; outline:none;"><tt>A<sub>1</sub></tt><div>\n'
        '<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        '    <td rowspan="2" style="line-height:100%"><pre>grblas.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</summary><div>\n"
        "<style scoped>\n"
        "    .dataframe tbody tr th:only-of-type {\n"
        "        vertical-align: middle;\n"
        "    }\n"
        "\n"
        "    .dataframe tbody tr th {\n"
        "        vertical-align: top;\n"
        "    }\n"
        "\n"
        "    .dataframe thead th {\n"
        "        text-align: right;\n"
        "    }\n"
        "</style>\n"
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        "      <th></th>\n"
        "      <th>0</th>\n"
        "      <th>1</th>\n"
        "      <th>2</th>\n"
        "      <th>3</th>\n"
        "      <th>4</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>0</td>\n"
        "      <td></td>\n"
        "      <td>1</td>\n"
        "      <td></td>\n"
        "      <td>2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>Do <code>expr.new()</code> or <code>other << expr</code> to calculate the expression.</em></div>"
    )


def test_matrix_reduce_repr(C, v):
    repr_printer(C.reduce_scalar(), "C.reduce_scalar()")
    assert repr(C.reduce_scalar()) == (
        "grblas.ScalarExpression              dtype\n"
        "C.reduce_scalar(monoid.plus[INT64])  INT64\n"
        "\n"
        "Do expr.new() or other << expr to calculate the expression."
    )


def test_matrix_reduce_repr_html(C, v):
    with pd.option_context("display.max_columns", 20):
        html_printer(C.reduce_scalar(), "C.reduce_scalar()", indent=8)
        assert repr_html(C.reduce_scalar()) == (
            '<div style="padding:4px;"><details><summary style="display:list-item; outline:none;"><b><tt>grblas.ScalarExpression:</tt></b><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>C.reduce_scalar(monoid.plus[INT64])</pre></td>\n'
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            '</summary><blockquote><div><details open><summary style="display:list-item; outline:none;"><tt>C</tt><div>\n'
            '<table style="border:1px solid black; max-width:100%;">\n'
            "  <tr>\n"
            '    <td rowspan="2" style="line-height:100%"><pre>grblas.Matrix</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>8</td>\n"
            "    <td>70</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><div>\n"
            "<style scoped>\n"
            "    .dataframe tbody tr th:only-of-type {\n"
            "        vertical-align: middle;\n"
            "    }\n"
            "\n"
            "    .dataframe tbody tr th {\n"
            "        vertical-align: top;\n"
            "    }\n"
            "\n"
            "    .dataframe thead th {\n"
            "        text-align: right;\n"
            "    }\n"
            "</style>\n"
            '<table border="1" class="dataframe">\n'
            "  <thead>\n"
            '    <tr style="text-align: right;">\n'
            "      <th></th>\n"
            "      <th>0</th>\n"
            "      <th>1</th>\n"
            "      <th>2</th>\n"
            "      <th>3</th>\n"
            "      <th>4</th>\n"
            "      <th>5</th>\n"
            "      <th>6</th>\n"
            "      <th>7</th>\n"
            "      <th>8</th>\n"
            "      <th>9</th>\n"
            "      <th>...</th>\n"
            "      <th>67</th>\n"
            "      <th>68</th>\n"
            "      <th>69</th>\n"
            "      <th>70</th>\n"
            "      <th>71</th>\n"
            "      <th>72</th>\n"
            "      <th>73</th>\n"
            "      <th>74</th>\n"
            "      <th>75</th>\n"
            "      <th>76</th>\n"
            "    </tr>\n"
            "  </thead>\n"
            "  <tbody>\n"
            "    <tr>\n"
            "      <th>0</th>\n"
            + "      <td></td>\n" * 4
            + "      <td>0</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>5</td>\n"
            + "      <td></td>\n" * 4
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>1</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>2</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>3</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>4</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>...</th>\n" + "      <td>...</td>\n" * 21 + "    </tr>\n"
            "    <tr>\n"
            "      <th>65</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>66</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>67</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>68</th>\n"
            + "      <td></td>\n" * 10
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 10
            + "    </tr>\n"
            "    <tr>\n"
            "      <th>69</th>\n"
            + "      <td></td>\n" * 4
            + "      <td>4</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>...</td>\n"
            + "      <td></td>\n" * 5
            + "      <td>8</td>\n"
            + "      <td></td>\n" * 4
            + "    </tr>\n"
            "  </tbody>\n"
            "</table>\n"
            "</div></details></div></blockquote></details><em>Do <code>expr.new()</code> or <code>other << expr</code> to calculate the expression.</em></div>"
        )
