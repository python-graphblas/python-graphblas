import numpy as np
import pytest

import graphblas
from graphblas import dtypes, formatting, unary
from graphblas.formatting import CSS_STYLE

from .conftest import autocompute

from graphblas import Matrix, Scalar, Vector  # isort:skip (for dask-graphblas)


try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


def repr_html(x):
    return x._repr_html_()


def _printer(text, name, repr_name, indent):
    to_print = []
    _print = to_print.append
    indent = " " * indent
    _print(f"{indent}assert {repr_name}({name}) == (")
    lines = text.split("\n")
    prev_line = ""
    count = 0
    in_style = False
    is_style = False
    for line in lines[:-1]:
        if in_style:
            if line.startswith("</style>"):
                # line = f"f'{{CSS_STYLE}}'"
                in_style = False
                is_style = True
            else:  # pragma: no cover
                # This definitely gets covered, but why is it not picked up?
                continue
        if repr_name == "repr_html" and line.startswith("<style>"):
            prev_line = prev_line[:-1]  # remove "\n"
            in_style = True
            continue

        line = line + "\n"
        if line == prev_line:
            count += 1
        else:
            if count == 1:
                _print(f"{indent}    {prev_line!r}")
            elif count > 1:
                if to_print[-1][-1] == "+":
                    _print(f"{indent}    {prev_line!r} * {count} +")
                else:
                    _print(f"{indent}  + {prev_line!r} * {count} +")
            if is_style:
                _print(f"{indent}   f'{{CSS_STYLE}}'")
                count = 0
                is_style = False
            else:
                count = 1
            prev_line = line
    if count == 1:
        _print(f"{indent}    {prev_line!r}")
    elif count > 1:  # pragma: no cover
        _print(f"{indent}    {prev_line!r} * {count}")
    _print(f"{indent}    {lines[-1]!r}")
    _print(f"{indent})")
    print("\n".join(to_print))


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
    return Scalar(int, name="t")


def test_no_pandas_repr(A, C, v, w):
    # This is a bit of a hack...
    has_pandas_prev = formatting.has_pandas
    formatting.has_pandas = False
    try:
        repr_printer(A, "A", indent=8)
        assert repr(A) == (
            '"A_1"      nvals  nrows  ncols  dtype   format\n'
            "gb.Matrix      3      1      5  INT64  bitmapr"
        )
        repr_printer(A.T, "A.T", indent=8)
        assert repr(A.T) == (
            '"A_1.T"              nvals  nrows  ncols  dtype   format\n'
            "gb.TransposedMatrix      3      5      1  INT64  bitmapc"
        )
        repr_printer(C.S, "C.S", indent=8)
        assert repr(C.S) == (
            '"C.S"           nvals  nrows  ncols  dtype    format\n'
            "StructuralMask\n"
            "of gb.Matrix        8     70     77  INT64  hypercsr"
        )
        repr_printer(v, "v", indent=8)
        assert repr(v) == (
            '"v"        nvals  size  dtype  format\n' "gb.Vector      3     5   FP64  bitmap"
        )
        repr_printer(~w.V, "~w.V", indent=8)
        assert repr(~w.V) == (
            '"~w.V"                 nvals  size  dtype  format\n'
            "ComplementedValueMask\n"
            "of gb.Vector               4    77  INT64  bitmap"
        )
    finally:
        formatting.has_pandas = has_pandas_prev


@pytest.mark.skipif("not pd")
def test_matrix_repr_small(A, B):
    repr_printer(A, "A")
    assert repr(A) == (
        '"A_1"      nvals  nrows  ncols  dtype   format\n'
        "gb.Matrix      3      1      5  INT64  bitmapr\n"
        "----------------------------------------------\n"
        "   0 1  2 3  4\n"
        "0  0    1    2"
    )
    repr_printer(B, "B")
    assert repr(B) == (
        '"B_1"      nvals  nrows  ncols  dtype   format\n'
        "gb.Matrix      3      5      1  INT64  bitmapc\n"
        "----------------------------------------------\n"
        "    0\n"
        "0  10\n"
        "1    \n"
        "2  20\n"
        "3    \n"
        "4  30"
    )
    repr_printer(B.T, "B.T")
    assert repr(B.T) == (
        '"B_1.T"              nvals  nrows  ncols  dtype   format\n'
        "gb.TransposedMatrix      3      1      5  INT64  bitmapr\n"
        "--------------------------------------------------------\n"
        "    0 1   2 3   4\n"
        "0  10    20    30"
    )


@pytest.mark.skipif("not pd")
def test_matrix_mask_repr_small(A):
    repr_printer(A.S, "A.S")
    assert repr(A.S) == (
        '"A_1.S"         nvals  nrows  ncols  dtype   format\n'
        "StructuralMask\n"
        "of gb.Matrix        3      1      5  INT64  bitmapr\n"
        "---------------------------------------------------\n"
        "   0 1  2 3  4\n"
        "0  1    1    1"
    )
    repr_printer(A.V, "A.V")
    assert repr(A.V) == (
        '"A_1.V"       nvals  nrows  ncols  dtype   format\n'
        "ValueMask   \n"
        "of gb.Matrix      3      1      5  INT64  bitmapr\n"
        "-------------------------------------------------\n"
        "   0 1  2 3  4\n"
        "0  0    1    1"
    )
    repr_printer(~A.S, "~A.S")
    assert repr(~A.S) == (
        '"~A_1.S"                    nvals  nrows  ncols  dtype   format\n'
        "ComplementedStructuralMask\n"
        "of gb.Matrix                    3      1      5  INT64  bitmapr\n"
        "---------------------------------------------------------------\n"
        "   0 1  2 3  4\n"
        "0  0    0    0"
    )
    repr_printer(~A.V, "~A.V")
    assert repr(~A.V) == (
        '"~A_1.V"               nvals  nrows  ncols  dtype   format\n'
        "ComplementedValueMask\n"
        "of gb.Matrix               3      1      5  INT64  bitmapr\n"
        "----------------------------------------------------------\n"
        "   0 1  2 3  4\n"
        "0  1    0    0"
    )


@pytest.mark.skipif("not pd")
def test_matrix_repr_large(C, D):
    with pd.option_context("display.max_columns", 24, "display.width", 100):
        repr_printer(C, "C", indent=8)
        assert repr(C) == (
            '"C"        nvals  nrows  ncols  dtype    format\n'
            "gb.Matrix      8     70     77  INT64  hypercsr\n"
            "-----------------------------------------------\n"
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
            '"C.T"                nvals  nrows  ncols  dtype    format\n'
            "gb.TransposedMatrix      8     77     70  INT64  hypercsc\n"
            "---------------------------------------------------------\n"
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
            '"D_skinny_in_one_dim"  nvals  nrows  ncols  dtype    format\n'
            "gb.Matrix                  4     70      5   BOOL  hypercsr\n"
            "-----------------------------------------------------------\n"
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
            '"D_skinny_in_one_dim.T"  nvals  nrows  ncols  dtype    format\n'
            "gb.TransposedMatrix          4      5     70   BOOL  hypercsc\n"
            "-------------------------------------------------------------\n"
            "     0  1  2  3  4  5  6  7  8      9  10 11  ... 58 59    60 61 62 63 64 65 66 67 68     69\n"
            "0                                             ...                                           \n"
            "1                                             ...                                           \n"
            "2                                             ...                                           \n"
            "3                                             ...                                           \n"
            "4  True                          False        ...        True                          False"
        )


@pytest.mark.skipif("not pd")
def test_matrix_mask_repr_large(C):
    with pd.option_context("display.max_columns", 24, "display.width", 100):
        repr_printer(C.S, "C.S", indent=8)
        assert repr(C.S) == (
            '"C.S"           nvals  nrows  ncols  dtype    format\n'
            "StructuralMask\n"
            "of gb.Matrix        8     70     77  INT64  hypercsr\n"
            "----------------------------------------------------\n"
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
            '"C.V"         nvals  nrows  ncols  dtype    format\n'
            "ValueMask   \n"
            "of gb.Matrix      8     70     77  INT64  hypercsr\n"
            "--------------------------------------------------\n"
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
            '"~C.S"                      nvals  nrows  ncols  dtype    format\n'
            "ComplementedStructuralMask\n"
            "of gb.Matrix                    8     70     77  INT64  hypercsr\n"
            "----------------------------------------------------------------\n"
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
            '"~C.V"                 nvals  nrows  ncols  dtype    format\n'
            "ComplementedValueMask\n"
            "of gb.Matrix               8     70     77  INT64  hypercsr\n"
            "-----------------------------------------------------------\n"
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


@pytest.mark.skipif("not pd")
def test_vector_repr_small(v):
    repr_printer(v, "v")
    assert repr(v) == (
        '"v"        nvals  size  dtype  format\n'
        "gb.Vector      3     5   FP64  bitmap\n"
        "-------------------------------------\n"
        "index    0 1    2 3    4\n"
        "value  0.0    1.1    2.2"
    )


@pytest.mark.skipif("not pd")
def test_vector_repr_large(w):
    with pd.option_context("display.max_columns", 26, "display.width", 100):
        repr_printer(w, "w", indent=8)
        assert repr(w) == (
            '"w"        nvals  size  dtype  format\n'
            "gb.Vector      4    77  INT64  bitmap\n"
            "-------------------------------------\n"
            "index 0  1  2  3  4  5  6  7  8  9  10 11 12  ... 64 65 66 67 68 69 70 71 72 73 74 75 76\n"
            "value  1              2                       ...  3              4                     "
        )


@pytest.mark.skipif("not pd")
def test_vector_mask_repr_small(v):
    repr_printer(v.S, "v.S")
    assert repr(v.S) == (
        '"v.S"           nvals  size  dtype  format\n'
        "StructuralMask\n"
        "of gb.Vector        3     5   FP64  bitmap\n"
        "------------------------------------------\n"
        "index  0 1  2 3  4\n"
        "value  1    1    1"
    )
    repr_printer(v.V, "v.V")
    assert repr(v.V) == (
        '"v.V"         nvals  size  dtype  format\n'
        "ValueMask   \n"
        "of gb.Vector      3     5   FP64  bitmap\n"
        "----------------------------------------\n"
        "index  0 1  2 3  4\n"
        "value  0    1    1"
    )
    repr_printer(~v.S, "~v.S")
    assert repr(~v.S) == (
        '"~v.S"                      nvals  size  dtype  format\n'
        "ComplementedStructuralMask\n"
        "of gb.Vector                    3     5   FP64  bitmap\n"
        "------------------------------------------------------\n"
        "index  0 1  2 3  4\n"
        "value  0    0    0"
    )
    repr_printer(~v.V, "~v.V")
    assert repr(~v.V) == (
        '"~v.V"                 nvals  size  dtype  format\n'
        "ComplementedValueMask\n"
        "of gb.Vector               3     5   FP64  bitmap\n"
        "-------------------------------------------------\n"
        "index  0 1  2 3  4\n"
        "value  1    0    0"
    )


@pytest.mark.skipif("not pd")
def test_vector_mask_repr_large(w):
    with pd.option_context("display.max_columns", 26, "display.width", 100):
        repr_printer(w.S, "w.S", indent=8)
        assert repr(w.S) == (
            '"w.S"           nvals  size  dtype  format\n'
            "StructuralMask\n"
            "of gb.Vector        4    77  INT64  bitmap\n"
            "------------------------------------------\n"
            "index 0  1  2  3  4  5  6  7  8  9  10 11 12  ... 64 65 66 67 68 69 70 71 72 73 74 75 76\n"
            "value  1              1                       ...  1              1                     "
        )
        repr_printer(w.V, "w.V", indent=8)
        assert repr(w.V) == (
            '"w.V"         nvals  size  dtype  format\n'
            "ValueMask   \n"
            "of gb.Vector      4    77  INT64  bitmap\n"
            "----------------------------------------\n"
            "index 0  1  2  3  4  5  6  7  8  9  10 11 12  ... 64 65 66 67 68 69 70 71 72 73 74 75 76\n"
            "value  1              1                       ...  1              1                     "
        )
        repr_printer(~w.S, "~w.S", indent=8)
        assert repr(~w.S) == (
            '"~w.S"                      nvals  size  dtype  format\n'
            "ComplementedStructuralMask\n"
            "of gb.Vector                    4    77  INT64  bitmap\n"
            "------------------------------------------------------\n"
            "index 0  1  2  3  4  5  6  7  8  9  10 11 12  ... 64 65 66 67 68 69 70 71 72 73 74 75 76\n"
            "value  0              0                       ...  0              0                     "
        )
        repr_printer(~w.V, "~w.V", indent=8)
        assert repr(~w.V) == (
            '"~w.V"                 nvals  size  dtype  format\n'
            "ComplementedValueMask\n"
            "of gb.Vector               4    77  INT64  bitmap\n"
            "-------------------------------------------------\n"
            "index 0  1  2  3  4  5  6  7  8  9  10 11 12  ... 64 65 66 67 68 69 70 71 72 73 74 75 76\n"
            "value  0              0                       ...  0              0                     "
        )


def test_scalar_repr(s, t):
    repr_printer(s, "s")
    assert repr(s) == ('"s_1"      value  dtype\n' "gb.Scalar     42  INT64")
    assert repr(t) == ('"t"        value  dtype\n' "gb.Scalar   None  INT64")


def test_no_pandas_repr_html(A, C, v, w):
    # This is a bit of a hack...
    has_pandas_prev = formatting.has_pandas
    formatting.has_pandas = False
    try:
        html_printer(A, "A", indent=8)
        assert repr_html(A) == (
            "<div>"
            f"{CSS_STYLE}"
            '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub></tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>3</td>\n"
            "    <td>1</td>\n"
            "    <td>5</td>\n"
            "    <td>INT64</td>\n"
            "    <td>bitmapr</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><em>(Install</em> <tt>pandas</tt> <em>to see a preview of the data)</em></details></div>"
        )
        html_printer(A.T, "A.T", indent=8)
        assert repr_html(A.T) == (
            "<div>"
            f"{CSS_STYLE}"
            '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub>.T</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.TransposedMatrix</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>3</td>\n"
            "    <td>5</td>\n"
            "    <td>1</td>\n"
            "    <td>INT64</td>\n"
            "    <td>bitmapc</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><em>(Install</em> <tt>pandas</tt> <em>to see a preview of the data)</em></details></div>"
        )
        html_printer(C.S, "C.S", indent=8)
        assert repr_html(C.S) == (
            "<div>"
            f"{CSS_STYLE}"
            '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>C.S</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>StructuralMask\n'
            "of\n"
            "gb.Matrix</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>8</td>\n"
            "    <td>70</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "    <td>hypercsr</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><em>(Install</em> <tt>pandas</tt> <em>to see a preview of the data)</em></details></div>"
        )
        html_printer(v, "v", indent=8)
        assert repr_html(v) == (
            "<div>"
            f"{CSS_STYLE}"
            '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>size</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>3</td>\n"
            "    <td>5</td>\n"
            "    <td>FP64</td>\n"
            "    <td>bitmap</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><em>(Install</em> <tt>pandas</tt> <em>to see a preview of the data)</em></details></div>"
        )
        html_printer(~w.V, "~w.V", indent=8)
        assert repr_html(~w.V) == (
            "<div>"
            f"{CSS_STYLE}"
            '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>~w.V</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>ComplementedValueMask\n'
            "of\n"
            "gb.Vector</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>size</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>4</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "    <td>bitmap</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            "</summary><em>(Install</em> <tt>pandas</tt> <em>to see a preview of the data)</em></details></div>"
        )
    finally:
        formatting.has_pandas = has_pandas_prev


@pytest.mark.skipif("not pd")
def test_matrix_repr_html_small(A, B):
    html_printer(A, "A")
    assert repr_html(A) == (
        "<div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "<div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>B<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>1</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapc</td>\n"
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
        "<div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>B<sub>1</sub>.T</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.TransposedMatrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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


@pytest.mark.skipif("not pd")
def test_matrix_mask_repr_html_small(A):
    html_printer(A.S, "A.S")
    assert repr_html(A.S) == (
        "<div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub>.S</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>StructuralMask\n'
        "of\n"
        "gb.Matrix</pre></td>\n"
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "<div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub>.V</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>ValueMask\n'
        "of\n"
        "gb.Matrix</pre></td>\n"
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "<div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>~A<sub>1</sub>.S</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>ComplementedStructuralMask\n'
        "of\n"
        "gb.Matrix</pre></td>\n"
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "<div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>~A<sub>1</sub>.V</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>ComplementedValueMask\n'
        "of\n"
        "gb.Matrix</pre></td>\n"
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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


@pytest.mark.skipif("not pd")
def test_matrix_repr_html_large(C, D):
    with pd.option_context("display.max_columns", 20):
        html_printer(C, "C", indent=8)
        assert repr_html(C) == (
            "<div>"
            f"{CSS_STYLE}"
            '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>C</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>8</td>\n"
            "    <td>70</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "    <td>hypercsr</td>\n"
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
            "<div>"
            f"{CSS_STYLE}"
            '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>C.T</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.TransposedMatrix</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>8</td>\n"
            "    <td>77</td>\n"
            "    <td>70</td>\n"
            "    <td>INT64</td>\n"
            "    <td>hypercsc</td>\n"
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
            "<div>"
            f"{CSS_STYLE}"
            '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>D<sub>skinny_in_one_dim</sub></tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>4</td>\n"
            "    <td>70</td>\n"
            "    <td>5</td>\n"
            "    <td>BOOL</td>\n"
            "    <td>hypercsr</td>\n"
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
            "<div>"
            f"{CSS_STYLE}"
            '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>D<sub>skinny_in_one_dim</sub>.T</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.TransposedMatrix</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>4</td>\n"
            "    <td>5</td>\n"
            "    <td>70</td>\n"
            "    <td>BOOL</td>\n"
            "    <td>hypercsc</td>\n"
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


@pytest.mark.skipif("not pd")
def test_matrix_mask_repr_html_large(C):
    with pd.option_context("display.max_columns", 20):
        html_printer(C.S, "C.S", indent=8)
        assert repr_html(C.S) == (
            "<div>"
            f"{CSS_STYLE}"
            '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>C.S</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>StructuralMask\n'
            "of\n"
            "gb.Matrix</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>8</td>\n"
            "    <td>70</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "    <td>hypercsr</td>\n"
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
            "<div>"
            f"{CSS_STYLE}"
            '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>C.V</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>ValueMask\n'
            "of\n"
            "gb.Matrix</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>8</td>\n"
            "    <td>70</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "    <td>hypercsr</td>\n"
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
            "<div>"
            f"{CSS_STYLE}"
            '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>~C.S</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>ComplementedStructuralMask\n'
            "of\n"
            "gb.Matrix</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>8</td>\n"
            "    <td>70</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "    <td>hypercsr</td>\n"
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
            "<div>"
            f"{CSS_STYLE}"
            '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>~C.V</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>ComplementedValueMask\n'
            "of\n"
            "gb.Matrix</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>8</td>\n"
            "    <td>70</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "    <td>hypercsr</td>\n"
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


@pytest.mark.skipif("not pd")
def test_vector_repr_html_small(v):
    html_printer(v, "v")
    assert repr_html(v) == (
        "<div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )


@pytest.mark.skipif("not pd")
def test_vector_repr_html_large(w):
    with pd.option_context("display.max_columns", 20):
        html_printer(w, "w", indent=8)
        assert repr_html(w) == (
            "<div>"
            f"{CSS_STYLE}"
            '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>w</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>size</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>4</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "    <td>bitmap</td>\n"
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


@pytest.mark.skipif("not pd")
def test_vector_mask_repr_html_small(v):
    html_printer(v.S, "v.S")
    assert repr_html(v.S) == (
        "<div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>v.S</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>StructuralMask\n'
        "of\n"
        "gb.Vector</pre></td>\n"
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "<div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>v.V</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>ValueMask\n'
        "of\n"
        "gb.Vector</pre></td>\n"
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "<div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>~v.S</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>ComplementedStructuralMask\n'
        "of\n"
        "gb.Vector</pre></td>\n"
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "<div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>~v.V</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>ComplementedValueMask\n'
        "of\n"
        "gb.Vector</pre></td>\n"
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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


@pytest.mark.skipif("not pd")
def test_vector_mask_repr_html_large(w):
    with pd.option_context("display.max_columns", 20):
        html_printer(w.S, "w.S", indent=8)
        assert repr_html(w.S) == (
            "<div>"
            f"{CSS_STYLE}"
            '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>w.S</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>StructuralMask\n'
            "of\n"
            "gb.Vector</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>size</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>4</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "    <td>bitmap</td>\n"
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
            "<div>"
            f"{CSS_STYLE}"
            '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>w.V</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>ValueMask\n'
            "of\n"
            "gb.Vector</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>size</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>4</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "    <td>bitmap</td>\n"
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
            "<div>"
            f"{CSS_STYLE}"
            '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>~w.S</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>ComplementedStructuralMask\n'
            "of\n"
            "gb.Vector</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>size</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>4</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "    <td>bitmap</td>\n"
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
            "<div>"
            f"{CSS_STYLE}"
            '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>~w.V</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>ComplementedValueMask\n'
            "of\n"
            "gb.Vector</pre></td>\n"
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>size</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>4</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "    <td>bitmap</td>\n"
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
        '<div class="gb-scalar"><tt>s<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Scalar</pre></td>\n'
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
        '<div class="gb-scalar"><tt>t</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Scalar</pre></td>\n'
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
        "gb.VectorExpression       size  dtype\n"
        "v.apply(unary.one[FP64])     5   FP64\n"
        "\n"
        "Do expr.new() or other << expr to calculate the expression."
    )


@pytest.mark.skipif("not pd")
def test_apply_repr_html(v):
    html_printer(v.apply(unary.one), "v.apply(unary.one)")
    assert repr_html(v.apply(unary.one)) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.VectorExpression:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>v.apply(unary.one[FP64])</pre></td>\n'
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
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
        "gb.MatrixExpression                          nrows  ncols  dtype\n"
        "A_1.mxm(B_1, op=semiring.plus_times[INT64])      1      1  INT64\n"
        "\n"
        "Do expr.new() or other << expr to calculate the expression."
    )


@pytest.mark.skipif("not pd")
def test_mxm_repr_html(A, B):
    html_printer(A.mxm(B), "A.mxm(B)")
    assert repr_html(A.mxm(B)) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.MatrixExpression:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>A<sub>1</sub>.mxm(B<sub>1</sub>, op=semiring.plus_times[INT64])</pre></td>\n'
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n" + "    <td>1</td>\n" * 2 + "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>B<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>1</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapc</td>\n"
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
        "gb.VectorExpression                       size  dtype\n"
        "A_1.mxv(v, op=semiring.plus_times[FP64])     1   FP64\n"
        "\n"
        "Do expr.new() or other << expr to calculate the expression."
    )


@pytest.mark.skipif("not pd")
def test_mxv_repr_html(A, v):
    html_printer(A.mxv(v), "A.mxv(v)")
    assert repr_html(A.mxv(v)) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.VectorExpression:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>A<sub>1</sub>.mxv(v, op=semiring.plus_times[FP64])</pre></td>\n'
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>1</td>\n"
        "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>Do <code>expr.new()</code> or <code>other << expr</code> to calculate the expression.</em></div>"
    )


@pytest.mark.skipif("not pd")
def test_matrix_reduce_columns_repr_html(A):
    # This is implemented using the transpose of A, so make sure we're oriented correctly!
    html_printer(A.reduce_columnwise(), "A.reduce_columnwise()")
    assert repr_html(A.reduce_columnwise()) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.VectorExpression:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>A<sub>1</sub>.reduce_columnwise(monoid.plus[INT64])</pre></td>\n'
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "gb.ScalarExpression                  dtype\n"
        "C.reduce_scalar(monoid.plus[INT64])  INT64\n"
        "\n"
        "Do expr.new() or other << expr to calculate the expression."
    )


@pytest.mark.skipif("not pd")
def test_matrix_reduce_repr_html(C, v):
    with pd.option_context("display.max_columns", 20):
        html_printer(C.reduce_scalar(), "C.reduce_scalar()", indent=8)
        assert repr_html(C.reduce_scalar()) == (
            '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.ScalarExpression:</tt></b><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>C.reduce_scalar(monoid.plus[INT64])</pre></td>\n'
            "    <td><pre>dtype</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>INT64</td>\n"
            "  </tr>\n"
            "</table>\n"
            "</div>\n"
            '</summary><blockquote class="gb-expr-blockquote"><div>'
            f"{CSS_STYLE}"
            '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>C</tt><div>\n'
            '<table class="gb-info-table">\n'
            "  <tr>\n"
            '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
            "    <td><pre>nvals</pre></td>\n"
            "    <td><pre>nrows</pre></td>\n"
            "    <td><pre>ncols</pre></td>\n"
            "    <td><pre>dtype</pre></td>\n"
            "    <td><pre>format</pre></td>\n"
            "  </tr>\n"
            "  <tr>\n"
            "    <td>8</td>\n"
            "    <td>70</td>\n"
            "    <td>77</td>\n"
            "    <td>INT64</td>\n"
            "    <td>hypercsr</td>\n"
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


@pytest.mark.skipif("not pd")
def test_matrix_huge():
    M = Matrix(int, nrows=2**60, ncols=2**60, name="M")
    repr_printer(M, "M")
    assert repr(M) == (
        '"M"        nvals                nrows                ncols  dtype    format\n'
        "gb.Matrix      0  1152921504606846976  1152921504606846976  INT64  hypercsr\n"
        "---------------------------------------------------------------------------\n"
        "                    0                    ... 1152921504606846975\n"
        "0                                        ...                    \n"
        "1                                        ...                    \n"
        "2                                        ...                    \n"
        "3                                        ...                    \n"
        "4                                        ...                    \n"
        "...                                 ...  ...                 ...\n"
        "1152921504606846971                      ...                    \n"
        "1152921504606846972                      ...                    \n"
        "1152921504606846973                      ...                    \n"
        "1152921504606846974                      ...                    \n"
        "1152921504606846975                      ...                    "
    )
    assert 2**60 - 1 == 1152921504606846975  # sanity
    M2 = M[0:, 0:].new()
    assert M.isequal(M2)


@pytest.mark.skipif("not pd")
def test_matrix_huge_html():
    M = Matrix(int, nrows=2**60, ncols=2**60, name="M")
    html_printer(M, "M")
    assert repr_html(M) == (
        "<div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>M</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>0</td>\n" + "    <td>1152921504606846976</td>\n" * 2 + "    <td>INT64</td>\n"
        "    <td>hypercsr</td>\n"
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
        "      <th>10</th>\n"
        "      <th>11</th>\n"
        "      <th>12</th>\n"
        "      <th>13</th>\n"
        "      <th>14</th>\n"
        "      <th>15</th>\n"
        "      <th>16</th>\n"
        "      <th>17</th>\n"
        "      <th>18</th>\n"
        "      <th>19</th>\n"
        "      <th>20</th>\n"
        "      <th>21</th>\n"
        "      <th>22</th>\n"
        "      <th>23</th>\n"
        "      <th>24</th>\n"
        "      <th>25</th>\n"
        "      <th>26</th>\n"
        "      <th>27</th>\n"
        "      <th>28</th>\n"
        "      <th>29</th>\n"
        "      <th>30</th>\n"
        "      <th>31</th>\n"
        "      <th>32</th>\n"
        "      <th>33</th>\n"
        "      <th>34</th>\n"
        "      <th>35</th>\n"
        "      <th>36</th>\n"
        "      <th>37</th>\n"
        "      <th>38</th>\n"
        "      <th>39</th>\n"
        "      <th>...</th>\n"
        "      <th>1152921504606846936</th>\n"
        "      <th>1152921504606846937</th>\n"
        "      <th>1152921504606846938</th>\n"
        "      <th>1152921504606846939</th>\n"
        "      <th>1152921504606846940</th>\n"
        "      <th>1152921504606846941</th>\n"
        "      <th>1152921504606846942</th>\n"
        "      <th>1152921504606846943</th>\n"
        "      <th>1152921504606846944</th>\n"
        "      <th>1152921504606846945</th>\n"
        "      <th>1152921504606846946</th>\n"
        "      <th>1152921504606846947</th>\n"
        "      <th>1152921504606846948</th>\n"
        "      <th>1152921504606846949</th>\n"
        "      <th>1152921504606846950</th>\n"
        "      <th>1152921504606846951</th>\n"
        "      <th>1152921504606846952</th>\n"
        "      <th>1152921504606846953</th>\n"
        "      <th>1152921504606846954</th>\n"
        "      <th>1152921504606846955</th>\n"
        "      <th>1152921504606846956</th>\n"
        "      <th>1152921504606846957</th>\n"
        "      <th>1152921504606846958</th>\n"
        "      <th>1152921504606846959</th>\n"
        "      <th>1152921504606846960</th>\n"
        "      <th>1152921504606846961</th>\n"
        "      <th>1152921504606846962</th>\n"
        "      <th>1152921504606846963</th>\n"
        "      <th>1152921504606846964</th>\n"
        "      <th>1152921504606846965</th>\n"
        "      <th>1152921504606846966</th>\n"
        "      <th>1152921504606846967</th>\n"
        "      <th>1152921504606846968</th>\n"
        "      <th>1152921504606846969</th>\n"
        "      <th>1152921504606846970</th>\n"
        "      <th>1152921504606846971</th>\n"
        "      <th>1152921504606846972</th>\n"
        "      <th>1152921504606846973</th>\n"
        "      <th>1152921504606846974</th>\n"
        "      <th>1152921504606846975</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        + "      <td></td>\n" * 40
        + "      <td>...</td>\n"
        + "      <td></td>\n" * 40
        + "    </tr>\n"
        "    <tr>\n"
        "      <th>1</th>\n"
        + "      <td></td>\n" * 40
        + "      <td>...</td>\n"
        + "      <td></td>\n" * 40
        + "    </tr>\n"
        "    <tr>\n"
        "      <th>2</th>\n"
        + "      <td></td>\n" * 40
        + "      <td>...</td>\n"
        + "      <td></td>\n" * 40
        + "    </tr>\n"
        "    <tr>\n"
        "      <th>3</th>\n"
        + "      <td></td>\n" * 40
        + "      <td>...</td>\n"
        + "      <td></td>\n" * 40
        + "    </tr>\n"
        "    <tr>\n"
        "      <th>4</th>\n"
        + "      <td></td>\n" * 40
        + "      <td>...</td>\n"
        + "      <td></td>\n" * 40
        + "    </tr>\n"
        "    <tr>\n"
        "      <th>...</th>\n" + "      <td>...</td>\n" * 81 + "    </tr>\n"
        "    <tr>\n"
        "      <th>1152921504606846971</th>\n"
        + "      <td></td>\n" * 40
        + "      <td>...</td>\n"
        + "      <td></td>\n" * 40
        + "    </tr>\n"
        "    <tr>\n"
        "      <th>1152921504606846972</th>\n"
        + "      <td></td>\n" * 40
        + "      <td>...</td>\n"
        + "      <td></td>\n" * 40
        + "    </tr>\n"
        "    <tr>\n"
        "      <th>1152921504606846973</th>\n"
        + "      <td></td>\n" * 40
        + "      <td>...</td>\n"
        + "      <td></td>\n" * 40
        + "    </tr>\n"
        "    <tr>\n"
        "      <th>1152921504606846974</th>\n"
        + "      <td></td>\n" * 40
        + "      <td>...</td>\n"
        + "      <td></td>\n" * 40
        + "    </tr>\n"
        "    <tr>\n"
        "      <th>1152921504606846975</th>\n"
        + "      <td></td>\n" * 40
        + "      <td>...</td>\n"
        + "      <td></td>\n" * 40
        + "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )


@pytest.mark.skipif("not pd")
def test_vector_huge():
    v = Vector(int, size=2**60)
    repr_printer(v, "v")
    assert repr(v) == (
        '"v_0"      nvals                 size  dtype  format\n'
        "gb.Vector      0  1152921504606846976  INT64  sparse\n"
        "----------------------------------------------------\n"
        "index 0                    ... 1152921504606846975\n"
        "value                      ...                    "
    )
    v2 = v[0:].new()
    assert v2.isequal(v)


@pytest.mark.skipif("not pd")
def test_vector_huge_html():
    v = Vector(int, size=2**60)
    html_printer(v, "v")
    assert repr_html(v) == (
        "<div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>v<sub>0</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>0</td>\n"
        "    <td>1152921504606846976</td>\n"
        "    <td>INT64</td>\n"
        "    <td>sparse</td>\n"
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
        "      <th>10</th>\n"
        "      <th>11</th>\n"
        "      <th>12</th>\n"
        "      <th>13</th>\n"
        "      <th>14</th>\n"
        "      <th>15</th>\n"
        "      <th>16</th>\n"
        "      <th>17</th>\n"
        "      <th>18</th>\n"
        "      <th>19</th>\n"
        "      <th>20</th>\n"
        "      <th>21</th>\n"
        "      <th>22</th>\n"
        "      <th>23</th>\n"
        "      <th>24</th>\n"
        "      <th>25</th>\n"
        "      <th>26</th>\n"
        "      <th>27</th>\n"
        "      <th>28</th>\n"
        "      <th>29</th>\n"
        "      <th>30</th>\n"
        "      <th>31</th>\n"
        "      <th>32</th>\n"
        "      <th>33</th>\n"
        "      <th>34</th>\n"
        "      <th>35</th>\n"
        "      <th>36</th>\n"
        "      <th>37</th>\n"
        "      <th>38</th>\n"
        "      <th>39</th>\n"
        "      <th>...</th>\n"
        "      <th>1152921504606846936</th>\n"
        "      <th>1152921504606846937</th>\n"
        "      <th>1152921504606846938</th>\n"
        "      <th>1152921504606846939</th>\n"
        "      <th>1152921504606846940</th>\n"
        "      <th>1152921504606846941</th>\n"
        "      <th>1152921504606846942</th>\n"
        "      <th>1152921504606846943</th>\n"
        "      <th>1152921504606846944</th>\n"
        "      <th>1152921504606846945</th>\n"
        "      <th>1152921504606846946</th>\n"
        "      <th>1152921504606846947</th>\n"
        "      <th>1152921504606846948</th>\n"
        "      <th>1152921504606846949</th>\n"
        "      <th>1152921504606846950</th>\n"
        "      <th>1152921504606846951</th>\n"
        "      <th>1152921504606846952</th>\n"
        "      <th>1152921504606846953</th>\n"
        "      <th>1152921504606846954</th>\n"
        "      <th>1152921504606846955</th>\n"
        "      <th>1152921504606846956</th>\n"
        "      <th>1152921504606846957</th>\n"
        "      <th>1152921504606846958</th>\n"
        "      <th>1152921504606846959</th>\n"
        "      <th>1152921504606846960</th>\n"
        "      <th>1152921504606846961</th>\n"
        "      <th>1152921504606846962</th>\n"
        "      <th>1152921504606846963</th>\n"
        "      <th>1152921504606846964</th>\n"
        "      <th>1152921504606846965</th>\n"
        "      <th>1152921504606846966</th>\n"
        "      <th>1152921504606846967</th>\n"
        "      <th>1152921504606846968</th>\n"
        "      <th>1152921504606846969</th>\n"
        "      <th>1152921504606846970</th>\n"
        "      <th>1152921504606846971</th>\n"
        "      <th>1152921504606846972</th>\n"
        "      <th>1152921504606846973</th>\n"
        "      <th>1152921504606846974</th>\n"
        "      <th>1152921504606846975</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        + "      <td></td>\n" * 40
        + "      <td>...</td>\n"
        + "      <td></td>\n" * 40
        + "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )


@pytest.mark.skipif("not pd")
def test_sparse_vector_repr():
    v = Vector.from_values([100 * i for i in range(100)], [10 * i for i in range(100)], name="v")
    repr_printer(v, "v")
    assert repr(v) == (
        '"v"        nvals  size  dtype  format\n'
        "gb.Vector    100  9901  INT64  sparse\n"
        "-------------------------------------\n"
        "    index  val\n"
        "0       0    0\n"
        "1     100   10\n"
        "2     200   20\n"
        "3     300   30\n"
        "4     400   40\n"
        "5     500   50\n"
        "6     600   60\n"
        "7     700   70\n"
        "8     800   80\n"
        "9     900   90\n"
        "...   ...  ..."
    )
    repr_printer(v.S, "v.S")
    assert repr(v.S) == (
        '"v.S"           nvals  size  dtype  format\n'
        "StructuralMask\n"
        "of gb.Vector      100  9901  INT64  sparse\n"
        "------------------------------------------\n"
        "    index  val\n"
        "0       0    1\n"
        "1     100    1\n"
        "2     200    1\n"
        "3     300    1\n"
        "4     400    1\n"
        "5     500    1\n"
        "6     600    1\n"
        "7     700    1\n"
        "8     800    1\n"
        "9     900    1\n"
        "...   ...  ..."
    )
    repr_printer(~v.S, "~v.S")
    assert repr(~v.S) == (
        '"~v.S"                      nvals  size  dtype  format\n'
        "ComplementedStructuralMask\n"
        "of gb.Vector                  100  9901  INT64  sparse\n"
        "------------------------------------------------------\n"
        "    index  val\n"
        "0       0    0\n"
        "1     100    0\n"
        "2     200    0\n"
        "3     300    0\n"
        "4     400    0\n"
        "5     500    0\n"
        "6     600    0\n"
        "7     700    0\n"
        "8     800    0\n"
        "9     900    0\n"
        "...   ...  ..."
    )
    repr_printer(v.V, "v.V")
    assert repr(v.V) == (
        '"v.V"         nvals  size  dtype  format\n'
        "ValueMask   \n"
        "of gb.Vector    100  9901  INT64  sparse\n"
        "----------------------------------------\n"
        "    index  val\n"
        "0     100    1\n"
        "1     200    1\n"
        "2     300    1\n"
        "3     400    1\n"
        "4     500    1\n"
        "5     600    1\n"
        "6     700    1\n"
        "7     800    1\n"
        "8     900    1\n"
        "9    1000    1\n"
        "...   ...  ..."
    )
    repr_printer(~v.V, "~v.V")
    assert repr(~v.V) == (
        '"~v.V"                 nvals  size  dtype  format\n'
        "ComplementedValueMask\n"
        "of gb.Vector             100  9901  INT64  sparse\n"
        "-------------------------------------------------\n"
        "    index  val\n"
        "0     100    0\n"
        "1     200    0\n"
        "2     300    0\n"
        "3     400    0\n"
        "4     500    0\n"
        "5     600    0\n"
        "6     700    0\n"
        "7     800    0\n"
        "8     900    0\n"
        "9    1000    0\n"
        "...   ...  ..."
    )
    v2 = v[:2000].new(name="v2")
    repr_printer(v2, "v2")
    assert repr(v2) == (
        '"v2"       nvals  size  dtype  format\n'
        "gb.Vector     20  2000  INT64  sparse\n"
        "-------------------------------------\n"
        "    index  val\n"
        "0       0    0\n"
        "1     100   10\n"
        "2     200   20\n"
        "3     300   30\n"
        "4     400   40\n"
        "5     500   50\n"
        "6     600   60\n"
        "7     700   70\n"
        "8     800   80\n"
        "9     900   90\n"
        "10   1000  100\n"
        "11   1100  110\n"
        "12   1200  120\n"
        "13   1300  130\n"
        "14   1400  140\n"
        "15   1500  150\n"
        "16   1600  160\n"
        "17   1700  170\n"
        "18   1800  180\n"
        "19   1900  190"
    )
    repr_printer(v2.V, "v2.V")
    assert repr(v2.V) == (
        '"v2.V"        nvals  size  dtype  format\n'
        "ValueMask   \n"
        "of gb.Vector     20  2000  INT64  sparse\n"
        "----------------------------------------\n"
        "    index  val\n"
        "0     100    1\n"
        "1     200    1\n"
        "2     300    1\n"
        "3     400    1\n"
        "4     500    1\n"
        "5     600    1\n"
        "6     700    1\n"
        "7     800    1\n"
        "8     900    1\n"
        "9    1000    1\n"
        "10   1100    1\n"
        "11   1200    1\n"
        "12   1300    1\n"
        "13   1400    1\n"
        "14   1500    1\n"
        "15   1600    1\n"
        "16   1700    1\n"
        "17   1800    1\n"
        "18   1900    1"
    )


@pytest.mark.skipif("not pd")
def test_sparse_matrix_repr():
    A = Matrix.from_values(
        [100 * i for i in range(100)], [10 * i for i in range(100)], list(range(100)), name="A"
    )
    repr_printer(A, "A")
    assert repr(A) == (
        '"A"        nvals  nrows  ncols  dtype    format\n'
        "gb.Matrix    100   9901    991  INT64  hypercsr\n"
        "-----------------------------------------------\n"
        "     row  col  val\n"
        "0      0    0    0\n"
        "1    100   10    1\n"
        "2    200   20    2\n"
        "3    300   30    3\n"
        "4    400   40    4\n"
        "5    500   50    5\n"
        "6    600   60    6\n"
        "7    700   70    7\n"
        "8    800   80    8\n"
        "9    900   90    9\n"
        "...  ...  ...  ..."
    )
    repr_printer(A.T, "A.T")
    assert repr(A.T) == (
        '"A.T"                nvals  nrows  ncols  dtype    format\n'
        "gb.TransposedMatrix    100    991   9901  INT64  hypercsc\n"
        "---------------------------------------------------------\n"
        "     row  col  val\n"
        "0      0    0    0\n"
        "1     10  100    1\n"
        "2     20  200    2\n"
        "3     30  300    3\n"
        "4     40  400    4\n"
        "5     50  500    5\n"
        "6     60  600    6\n"
        "7     70  700    7\n"
        "8     80  800    8\n"
        "9     90  900    9\n"
        "...  ...  ...  ..."
    )
    repr_printer(A.S, "A.S")
    assert repr(A.S) == (
        '"A.S"           nvals  nrows  ncols  dtype    format\n'
        "StructuralMask\n"
        "of gb.Matrix      100   9901    991  INT64  hypercsr\n"
        "----------------------------------------------------\n"
        "     row  col  val\n"
        "0      0    0    1\n"
        "1    100   10    1\n"
        "2    200   20    1\n"
        "3    300   30    1\n"
        "4    400   40    1\n"
        "5    500   50    1\n"
        "6    600   60    1\n"
        "7    700   70    1\n"
        "8    800   80    1\n"
        "9    900   90    1\n"
        "...  ...  ...  ..."
    )
    repr_printer(~A.S, "~A.S")
    assert repr(~A.S) == (
        '"~A.S"                      nvals  nrows  ncols  dtype    format\n'
        "ComplementedStructuralMask\n"
        "of gb.Matrix                  100   9901    991  INT64  hypercsr\n"
        "----------------------------------------------------------------\n"
        "     row  col  val\n"
        "0      0    0    0\n"
        "1    100   10    0\n"
        "2    200   20    0\n"
        "3    300   30    0\n"
        "4    400   40    0\n"
        "5    500   50    0\n"
        "6    600   60    0\n"
        "7    700   70    0\n"
        "8    800   80    0\n"
        "9    900   90    0\n"
        "...  ...  ...  ..."
    )
    repr_printer(A.V, "A.V")
    assert repr(A.V) == (
        '"A.V"         nvals  nrows  ncols  dtype    format\n'
        "ValueMask   \n"
        "of gb.Matrix    100   9901    991  INT64  hypercsr\n"
        "--------------------------------------------------\n"
        "      row  col  val\n"
        "0     100   10    1\n"
        "1     200   20    1\n"
        "2     300   30    1\n"
        "3     400   40    1\n"
        "4     500   50    1\n"
        "5     600   60    1\n"
        "6     700   70    1\n"
        "7     800   80    1\n"
        "8     900   90    1\n"
        "9    1000  100    1\n"
        "...   ...  ...  ..."
    )
    repr_printer(~A.V, "~A.V")
    assert repr(~A.V) == (
        '"~A.V"                 nvals  nrows  ncols  dtype    format\n'
        "ComplementedValueMask\n"
        "of gb.Matrix             100   9901    991  INT64  hypercsr\n"
        "-----------------------------------------------------------\n"
        "      row  col  val\n"
        "0     100   10    0\n"
        "1     200   20    0\n"
        "2     300   30    0\n"
        "3     400   40    0\n"
        "4     500   50    0\n"
        "5     600   60    0\n"
        "6     700   70    0\n"
        "7     800   80    0\n"
        "8     900   90    0\n"
        "9    1000  100    0\n"
        "...   ...  ...  ..."
    )
    A2 = A[:2000, :].new(name="A2")
    repr_printer(A2, "A2")
    assert repr(A2) == (
        '"A2"       nvals  nrows  ncols  dtype    format\n'
        "gb.Matrix     20   2000    991  INT64  hypercsr\n"
        "-----------------------------------------------\n"
        "     row  col  val\n"
        "0      0    0    0\n"
        "1    100   10    1\n"
        "2    200   20    2\n"
        "3    300   30    3\n"
        "4    400   40    4\n"
        "5    500   50    5\n"
        "6    600   60    6\n"
        "7    700   70    7\n"
        "8    800   80    8\n"
        "9    900   90    9\n"
        "10  1000  100   10\n"
        "11  1100  110   11\n"
        "12  1200  120   12\n"
        "13  1300  130   13\n"
        "14  1400  140   14\n"
        "15  1500  150   15\n"
        "16  1600  160   16\n"
        "17  1700  170   17\n"
        "18  1800  180   18\n"
        "19  1900  190   19"
    )
    repr_printer(A2.V, "A2.V")
    assert repr(A2.V) == (
        '"A2.V"        nvals  nrows  ncols  dtype    format\n'
        "ValueMask   \n"
        "of gb.Matrix     20   2000    991  INT64  hypercsr\n"
        "--------------------------------------------------\n"
        "     row  col  val\n"
        "0    100   10    1\n"
        "1    200   20    1\n"
        "2    300   30    1\n"
        "3    400   40    1\n"
        "4    500   50    1\n"
        "5    600   60    1\n"
        "6    700   70    1\n"
        "7    800   80    1\n"
        "8    900   90    1\n"
        "9   1000  100    1\n"
        "10  1100  110    1\n"
        "11  1200  120    1\n"
        "12  1300  130    1\n"
        "13  1400  140    1\n"
        "14  1500  150    1\n"
        "15  1600  160    1\n"
        "16  1700  170    1\n"
        "17  1800  180    1\n"
        "18  1900  190    1"
    )


@pytest.mark.skipif("not pd")
def test_infix_expr_repr_html(A, B, v):
    html_printer(v & v, "v & v")
    assert repr_html(v & v) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.VectorEwiseMultExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>v & v</pre></td>\n'
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>left_dtype</pre></td>\n"
        "    <td><pre>right_dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>5</td>\n" + "    <td>FP64</td>\n" * 2 + "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>Do <code>op(expr)</code> to create a <tt>VectorExpression</tt> for <tt>ewise_mult</tt>.<br>For example: <code>times(v & v)</code></em></div>"
    )
    html_printer(v | v, "v | v")
    assert repr_html(v | v) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.VectorEwiseAddExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>v | v</pre></td>\n'
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>left_dtype</pre></td>\n"
        "    <td><pre>right_dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>5</td>\n" + "    <td>FP64</td>\n" * 2 + "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>Do <code>op(expr)</code> to create a <tt>VectorExpression</tt> for <tt>ewise_add</tt>.<br>For example: <code>plus(v | v)</code></em></div>"
    )
    html_printer(A @ v, "A @ v")
    assert repr_html(A @ v) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.VectorMatMulExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>A<sub>1</sub> @ v</pre></td>\n'
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>left_dtype</pre></td>\n"
        "    <td><pre>right_dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>1</td>\n"
        "    <td>INT64</td>\n"
        "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>Do <code>op(expr)</code> to create a <tt>VectorExpression</tt> for <tt>mxv</tt>.<br>For example: <code>plus_times(A<sub>1</sub> @ v)</code></em></div>"
    )
    html_printer(v @ A.T, "v @ A.T")
    assert repr_html(v @ A.T) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.VectorMatMulExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>v @ A<sub>1</sub>.T</pre></td>\n'
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>left_dtype</pre></td>\n"
        "    <td><pre>right_dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>1</td>\n"
        "    <td>FP64</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub>.T</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.TransposedMatrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>1</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapc</td>\n"
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
        "      <td>0</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>1</th>\n"
        "      <td></td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>2</th>\n"
        "      <td>1</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>3</th>\n"
        "      <td></td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>4</th>\n"
        "      <td>2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>Do <code>op(expr)</code> to create a <tt>VectorExpression</tt> for <tt>vxm</tt>.<br>For example: <code>plus_times(v @ A<sub>1</sub>.T)</code></em></div>"
    )
    html_printer(A & B.T, "A & B.T")
    assert repr_html(A & B.T) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.MatrixEwiseMultExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>A<sub>1</sub> & B<sub>1</sub>.T</pre></td>\n'
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>right_dtype</pre></td>\n"
        "    <td><pre>left_dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n" + "    <td>INT64</td>\n" * 2 + "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>B<sub>1</sub>.T</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.TransposedMatrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "</div></details></div></blockquote></details><em>Do <code>op(expr)</code> to create a <tt>MatrixExpression</tt> for <tt>ewise_mult</tt>.<br>For example: <code>times(A<sub>1</sub> & B<sub>1</sub>.T)</code></em></div>"
    )
    html_printer(A | A, "A | A")
    assert repr_html(A | A) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.MatrixEwiseAddExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>A<sub>1</sub> | A<sub>1</sub></pre></td>\n'
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>right_dtype</pre></td>\n"
        "    <td><pre>left_dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n" + "    <td>INT64</td>\n" * 2 + "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "</div></details></div></blockquote></details><em>Do <code>op(expr)</code> to create a <tt>MatrixExpression</tt> for <tt>ewise_add</tt>.<br>For example: <code>plus(A<sub>1</sub> | A<sub>1</sub>)</code></em></div>"
    )
    html_printer(A @ B, "A @ B")
    assert repr_html(A @ B) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.MatrixMatMulExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>A<sub>1</sub> @ B<sub>1</sub></pre></td>\n'
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>right_dtype</pre></td>\n"
        "    <td><pre>left_dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n" + "    <td>1</td>\n" * 2 + "    <td>INT64</td>\n" * 2 + "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>B<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>1</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapc</td>\n"
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
        "</div></details></div></blockquote></details><em>Do <code>op(expr)</code> to create a <tt>MatrixExpression</tt> for <tt>mxm</tt>.<br>For example: <code>plus_times(A<sub>1</sub> @ B<sub>1</sub>)</code></em></div>"
    )
    html_printer(A.T @ B.T, "A.T @ B.T")
    assert repr_html(A.T @ B.T) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.MatrixMatMulExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>A<sub>1</sub>.T @ B<sub>1</sub>.T</pre></td>\n'
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>right_dtype</pre></td>\n"
        "    <td><pre>left_dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n" + "    <td>5</td>\n" * 2 + "    <td>INT64</td>\n" * 2 + "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub>.T</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.TransposedMatrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>1</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapc</td>\n"
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
        "      <td>0</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>1</th>\n"
        "      <td></td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>2</th>\n"
        "      <td>1</td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>3</th>\n"
        "      <td></td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>4</th>\n"
        "      <td>2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>B<sub>1</sub>.T</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.TransposedMatrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "</div></details></div></blockquote></details><em>Do <code>op(expr)</code> to create a <tt>MatrixExpression</tt> for <tt>mxm</tt>.<br>For example: <code>plus_times(A<sub>1</sub>.T @ B<sub>1</sub>.T)</code></em></div>"
    )
    html_printer(v @ v, "v @ v")
    assert repr_html(v @ v) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.ScalarMatMulExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>v @ v</pre></td>\n'
        "    <td><pre>left_dtype</pre></td>\n"
        "    <td><pre>right_dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n" + "    <td>FP64</td>\n" * 2 + "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>Do <code>op(expr)</code> to create a <tt>ScalarExpression</tt> for <tt>inner</tt>.<br>For example: <code>plus_times(v @ v)</code></em></div>"
    )


@pytest.mark.skipif("not pd")
def test_infix_expr_repr(A, B, v):
    repr_printer(v & v, "v & v")
    assert repr(v & v) == (
        "gb.VectorEwiseMultExpr  size  left_dtype  right_dtype\n"
        "v & v                      5        FP64         FP64\n"
        "\n"
        "Do op(expr) to create a VectorExpression for ewise_mult.\n"
        "For example: times(v & v)"
    )
    repr_printer(v | v, "v | v")
    assert repr(v | v) == (
        "gb.VectorEwiseAddExpr  size  left_dtype  right_dtype\n"
        "v | v                     5        FP64         FP64\n"
        "\n"
        "Do op(expr) to create a VectorExpression for ewise_add.\n"
        "For example: plus(v | v)"
    )
    repr_printer(A @ v, "A @ v")
    assert repr(A @ v) == (
        "gb.VectorMatMulExpr  size  left_dtype  right_dtype\n"
        "A_1 @ v                 1       INT64         FP64\n"
        "\n"
        "Do op(expr) to create a VectorExpression for mxv.\n"
        "For example: plus_times(A_1 @ v)"
    )
    repr_printer(v @ A.T, "v @ A.T")
    assert repr(v @ A.T) == (
        "gb.VectorMatMulExpr  size  left_dtype  right_dtype\n"
        "v @ A_1.T               1        FP64        INT64\n"
        "\n"
        "Do op(expr) to create a VectorExpression for vxm.\n"
        "For example: plus_times(v @ A_1.T)"
    )
    repr_printer(A & B.T, "A & B.T")
    assert repr(A & B.T) == (
        "gb.MatrixEwiseMultExpr  nrows  ncols  left_dtype  right_dtype\n"
        "A_1 & B_1.T                 1      5       INT64        INT64\n"
        "\n"
        "Do op(expr) to create a MatrixExpression for ewise_mult.\n"
        "For example: times(A_1 & B_1.T)"
    )
    repr_printer(A | A, "A | A")
    assert repr(A | A) == (
        "gb.MatrixEwiseAddExpr  nrows  ncols  left_dtype  right_dtype\n"
        "A_1 | A_1                  1      5       INT64        INT64\n"
        "\n"
        "Do op(expr) to create a MatrixExpression for ewise_add.\n"
        "For example: plus(A_1 | A_1)"
    )
    repr_printer(A @ B, "A @ B")
    assert repr(A @ B) == (
        "gb.MatrixMatMulExpr  nrows  ncols  left_dtype  right_dtype\n"
        "A_1 @ B_1                1      1       INT64        INT64\n"
        "\n"
        "Do op(expr) to create a MatrixExpression for mxm.\n"
        "For example: plus_times(A_1 @ B_1)"
    )
    repr_printer(A.T @ B.T, "A.T @ B.T")
    assert repr(A.T @ B.T) == (
        "gb.MatrixMatMulExpr  nrows  ncols  left_dtype  right_dtype\n"
        "A_1.T @ B_1.T            5      5       INT64        INT64\n"
        "\n"
        "Do op(expr) to create a MatrixExpression for mxm.\n"
        "For example: plus_times(A_1.T @ B_1.T)"
    )
    repr_printer(v @ v, "v @ v")
    assert repr(v @ v) == (
        "gb.ScalarMatMulExpr  left_dtype  right_dtype\n"
        "v @ v                      FP64         FP64\n"
        "\n"
        "Do op(expr) to create a ScalarExpression for inner.\n"
        "For example: plus_times(v @ v)"
    )


@pytest.mark.skipif("not pd")
def test_inner_outer_repr_html(v):
    html_printer(v.inner(v), "v.inner(v)")
    assert repr_html(v.inner(v)) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.ScalarExpression:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>v.inner(v, op=semiring.plus_times[FP64])</pre></td>\n'
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>Do <code>expr.new()</code> or <code>other << expr</code> to calculate the expression.</em></div>"
    )
    html_printer(v.outer(v), "v.outer(v)")
    assert repr_html(v.outer(v)) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.MatrixExpression:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>v.outer(v, op=semiring.any_times[FP64])</pre></td>\n'
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n" + "    <td>5</td>\n" * 2 + "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>Do <code>expr.new()</code> or <code>other << expr</code> to calculate the expression.</em></div>"
    )


@pytest.mark.skipif("not pd")
def test_inner_outer_repr(v):
    # XXX: hmm, having `(GrB_Matrix)` here isn't so pretty
    repr_printer(v.inner(v), "v.inner(v)")
    assert repr(v.inner(v)) == (
        "gb.ScalarExpression                                   dtype\n"
        "v.inner((GrB_Matrix)v, op=semiring.plus_times[FP64])   FP64\n"
        "\n"
        "Do expr.new() or other << expr to calculate the expression."
    )
    repr_printer(v.outer(v), "v.outer(v)")
    assert repr(v.outer(v)) == (
        "gb.MatrixExpression                                              nrows  ncols  dtype\n"
        "(GrB_Matrix)v.outer((GrB_Matrix)v, op=semiring.any_times[FP64])      5      5   FP64\n"
        "\n"
        "Do expr.new() or other << expr to calculate the expression."
    )


@autocompute
def test_autocompute(A, B, v):
    if not pd:  # pragma: no cover
        return
    repr_printer(A & A, "A & A")
    assert repr(A & A) == (
        "gb.MatrixEwiseMultExpr  nrows  ncols  left_dtype  right_dtype\n"
        "A_1 & A_1                   1      5       INT64        INT64\n"
        "\n"
        "Do op(expr) to create a MatrixExpression for ewise_mult.\n"
        "For example: times(A_1 & A_1)"
    )
    repr_printer(A.ewise_add(A), "A.ewise_add(A)")
    assert repr(A.ewise_add(A)) == (
        "gb.MatrixExpression                        nrows  ncols  dtype\n"
        "A_1.ewise_add(A_1, op=monoid.plus[INT64])      1      5  INT64\n"
        "\n"
        '"Result"   nvals  nrows  ncols  dtype   format\n'
        "gb.Matrix      3      1      5  INT64  bitmapr\n"
        "----------------------------------------------\n"
        "   0 1  2 3  4\n"
        "0  0    2    4\n"
        "\n"
        "Do expr.new() or other << expr to calculate the expression."
    )

    BIG = Vector(int, size=2**55)
    small = Vector(int, size=2**55)
    BIG[:] = 1
    small[0] = 2
    repr_printer(BIG.ewise_mult(small), "BIG.ewise_mult(small)")
    assert repr(BIG.ewise_mult(small)) == (
        "gb.VectorExpression                                       size  dtype\n"
        "v_0.ewise_mult(v_1, op=binary.times[INT64])  36028797018963968  INT64\n"
        "\n"
        '"Result"   nvals               size  dtype        format\n'
        "gb.Vector      1  36028797018963968  INT64  sparse (iso)\n"
        "--------------------------------------------------------\n"
        "index 0                 1                  ... 36028797018963966 36028797018963967\n"
        "value                 2                    ...                                    \n"
        "\n"
        "Do expr.new() or other << expr to calculate the expression."
    )
    repr_printer(BIG.ewise_add(small), "BIG.ewise_add(small)")
    assert repr(BIG.ewise_add(small)) == (
        "gb.VectorExpression                                     size  dtype\n"
        "v_0.ewise_add(v_1, op=monoid.plus[INT64])  36028797018963968  INT64\n"
        "\n"
        "Result is too large to compute!\n"
        "\n"
        "Do expr.new() or other << expr to calculate the expression."
    )
    BIG_bool = BIG.dup(dtype=bool)
    small_bool = small.dup(dtype=bool)
    small_bool[0] = False
    repr_printer(BIG_bool | small_bool, "BIG_bool | small_bool")
    assert repr(BIG_bool | small_bool) == (
        "gb.VectorEwiseAddExpr               size  left_dtype  right_dtype\n"
        "v_6 | v_7              36028797018963968        BOOL         BOOL\n"
        "\n"
        "Result is too large to compute!\n"
        "\n"
        "Do op(expr) to create a VectorExpression for ewise_add.\n"
        "For example: plus(v_6 | v_7)"
    )
    C = A.dup(dtype=bool)
    repr_printer(C & C, "C & C")
    assert repr(C & C) == (
        "gb.MatrixEwiseMultExpr  nrows  ncols  left_dtype  right_dtype\n"
        "M_2 & M_2                   1      5        BOOL         BOOL\n"
        "\n"
        '"Result"   nvals  nrows  ncols  dtype   format\n'
        "gb.Matrix      3      1      5   BOOL  bitmapr\n"
        "----------------------------------------------\n"
        "       0 1     2 3     4\n"
        "0  False    True    True\n"
        "\n"
        "Do op(expr) to create a MatrixExpression for ewise_mult.\n"
        "For example: times(M_2 & M_2)"
    )


@autocompute
def test_autocompute_html(A, B, v):
    if not pd:  # pragma: no cover
        return
    html_printer(A & A, "A & A")
    assert repr_html(A & A) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.MatrixEwiseMultExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>A<sub>1</sub> & A<sub>1</sub></pre></td>\n'
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>right_dtype</pre></td>\n"
        "    <td><pre>left_dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n" + "    <td>INT64</td>\n" * 2 + "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "</div></details></div></blockquote></details><em>Do <code>op(expr)</code> to create a <tt>MatrixExpression</tt> for <tt>ewise_mult</tt>.<br>For example: <code>times(A<sub>1</sub> & A<sub>1</sub>)</code></em></div>"
    )
    html_printer(A.ewise_add(A), "A.ewise_add(A)")
    assert repr_html(A.ewise_add(A)) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.MatrixExpression:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>A<sub>1</sub>.ewise_add(A<sub>1</sub>, op=monoid.plus[INT64])</pre></td>\n'
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "</div></details></div><hr><div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>Result</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "      <td>2</td>\n"
        "      <td></td>\n"
        "      <td>4</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>Do <code>expr.new()</code> or <code>other << expr</code> to calculate the expression.</em></div>"
    )

    BIG = Vector(int, size=2**55)
    small = Vector(int, size=2**55)
    BIG[:] = 1
    small[0] = 2
    html_printer(BIG.ewise_mult(small), "BIG.ewise_mult(small)")
    assert repr_html(BIG.ewise_mult(small)) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.VectorExpression:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>v<sub>0</sub>.ewise_mult(v<sub>1</sub>, op=binary.times[INT64])</pre></td>\n'
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>36028797018963968</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v<sub>0</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n" + "    <td>36028797018963968</td>\n" * 2 + "    <td>INT64</td>\n"
        "    <td>full (iso)</td>\n"
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
        "      <th>10</th>\n"
        "      <th>11</th>\n"
        "      <th>12</th>\n"
        "      <th>13</th>\n"
        "      <th>14</th>\n"
        "      <th>15</th>\n"
        "      <th>16</th>\n"
        "      <th>17</th>\n"
        "      <th>18</th>\n"
        "      <th>19</th>\n"
        "      <th>20</th>\n"
        "      <th>21</th>\n"
        "      <th>22</th>\n"
        "      <th>23</th>\n"
        "      <th>24</th>\n"
        "      <th>25</th>\n"
        "      <th>26</th>\n"
        "      <th>27</th>\n"
        "      <th>28</th>\n"
        "      <th>29</th>\n"
        "      <th>30</th>\n"
        "      <th>31</th>\n"
        "      <th>32</th>\n"
        "      <th>33</th>\n"
        "      <th>34</th>\n"
        "      <th>35</th>\n"
        "      <th>36</th>\n"
        "      <th>37</th>\n"
        "      <th>38</th>\n"
        "      <th>39</th>\n"
        "      <th>...</th>\n"
        "      <th>36028797018963928</th>\n"
        "      <th>36028797018963929</th>\n"
        "      <th>36028797018963930</th>\n"
        "      <th>36028797018963931</th>\n"
        "      <th>36028797018963932</th>\n"
        "      <th>36028797018963933</th>\n"
        "      <th>36028797018963934</th>\n"
        "      <th>36028797018963935</th>\n"
        "      <th>36028797018963936</th>\n"
        "      <th>36028797018963937</th>\n"
        "      <th>36028797018963938</th>\n"
        "      <th>36028797018963939</th>\n"
        "      <th>36028797018963940</th>\n"
        "      <th>36028797018963941</th>\n"
        "      <th>36028797018963942</th>\n"
        "      <th>36028797018963943</th>\n"
        "      <th>36028797018963944</th>\n"
        "      <th>36028797018963945</th>\n"
        "      <th>36028797018963946</th>\n"
        "      <th>36028797018963947</th>\n"
        "      <th>36028797018963948</th>\n"
        "      <th>36028797018963949</th>\n"
        "      <th>36028797018963950</th>\n"
        "      <th>36028797018963951</th>\n"
        "      <th>36028797018963952</th>\n"
        "      <th>36028797018963953</th>\n"
        "      <th>36028797018963954</th>\n"
        "      <th>36028797018963955</th>\n"
        "      <th>36028797018963956</th>\n"
        "      <th>36028797018963957</th>\n"
        "      <th>36028797018963958</th>\n"
        "      <th>36028797018963959</th>\n"
        "      <th>36028797018963960</th>\n"
        "      <th>36028797018963961</th>\n"
        "      <th>36028797018963962</th>\n"
        "      <th>36028797018963963</th>\n"
        "      <th>36028797018963964</th>\n"
        "      <th>36028797018963965</th>\n"
        "      <th>36028797018963966</th>\n"
        "      <th>36028797018963967</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        + "      <td>1</td>\n" * 40
        + "      <td>...</td>\n"
        + "      <td>1</td>\n" * 40
        + "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>1</td>\n"
        "    <td>36028797018963968</td>\n"
        "    <td>INT64</td>\n"
        "    <td>sparse (iso)</td>\n"
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
        "      <th>10</th>\n"
        "      <th>11</th>\n"
        "      <th>12</th>\n"
        "      <th>13</th>\n"
        "      <th>14</th>\n"
        "      <th>15</th>\n"
        "      <th>16</th>\n"
        "      <th>17</th>\n"
        "      <th>18</th>\n"
        "      <th>19</th>\n"
        "      <th>20</th>\n"
        "      <th>21</th>\n"
        "      <th>22</th>\n"
        "      <th>23</th>\n"
        "      <th>24</th>\n"
        "      <th>25</th>\n"
        "      <th>26</th>\n"
        "      <th>27</th>\n"
        "      <th>28</th>\n"
        "      <th>29</th>\n"
        "      <th>30</th>\n"
        "      <th>31</th>\n"
        "      <th>32</th>\n"
        "      <th>33</th>\n"
        "      <th>34</th>\n"
        "      <th>35</th>\n"
        "      <th>36</th>\n"
        "      <th>37</th>\n"
        "      <th>38</th>\n"
        "      <th>39</th>\n"
        "      <th>...</th>\n"
        "      <th>36028797018963928</th>\n"
        "      <th>36028797018963929</th>\n"
        "      <th>36028797018963930</th>\n"
        "      <th>36028797018963931</th>\n"
        "      <th>36028797018963932</th>\n"
        "      <th>36028797018963933</th>\n"
        "      <th>36028797018963934</th>\n"
        "      <th>36028797018963935</th>\n"
        "      <th>36028797018963936</th>\n"
        "      <th>36028797018963937</th>\n"
        "      <th>36028797018963938</th>\n"
        "      <th>36028797018963939</th>\n"
        "      <th>36028797018963940</th>\n"
        "      <th>36028797018963941</th>\n"
        "      <th>36028797018963942</th>\n"
        "      <th>36028797018963943</th>\n"
        "      <th>36028797018963944</th>\n"
        "      <th>36028797018963945</th>\n"
        "      <th>36028797018963946</th>\n"
        "      <th>36028797018963947</th>\n"
        "      <th>36028797018963948</th>\n"
        "      <th>36028797018963949</th>\n"
        "      <th>36028797018963950</th>\n"
        "      <th>36028797018963951</th>\n"
        "      <th>36028797018963952</th>\n"
        "      <th>36028797018963953</th>\n"
        "      <th>36028797018963954</th>\n"
        "      <th>36028797018963955</th>\n"
        "      <th>36028797018963956</th>\n"
        "      <th>36028797018963957</th>\n"
        "      <th>36028797018963958</th>\n"
        "      <th>36028797018963959</th>\n"
        "      <th>36028797018963960</th>\n"
        "      <th>36028797018963961</th>\n"
        "      <th>36028797018963962</th>\n"
        "      <th>36028797018963963</th>\n"
        "      <th>36028797018963964</th>\n"
        "      <th>36028797018963965</th>\n"
        "      <th>36028797018963966</th>\n"
        "      <th>36028797018963967</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        "      <td>2</td>\n"
        + "      <td></td>\n" * 39
        + "      <td>...</td>\n"
        + "      <td></td>\n" * 40
        + "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div><hr><div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>Result</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>1</td>\n"
        "    <td>36028797018963968</td>\n"
        "    <td>INT64</td>\n"
        "    <td>sparse (iso)</td>\n"
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
        "      <th>10</th>\n"
        "      <th>11</th>\n"
        "      <th>12</th>\n"
        "      <th>13</th>\n"
        "      <th>14</th>\n"
        "      <th>15</th>\n"
        "      <th>16</th>\n"
        "      <th>17</th>\n"
        "      <th>18</th>\n"
        "      <th>19</th>\n"
        "      <th>20</th>\n"
        "      <th>21</th>\n"
        "      <th>22</th>\n"
        "      <th>23</th>\n"
        "      <th>24</th>\n"
        "      <th>25</th>\n"
        "      <th>26</th>\n"
        "      <th>27</th>\n"
        "      <th>28</th>\n"
        "      <th>29</th>\n"
        "      <th>30</th>\n"
        "      <th>31</th>\n"
        "      <th>32</th>\n"
        "      <th>33</th>\n"
        "      <th>34</th>\n"
        "      <th>35</th>\n"
        "      <th>36</th>\n"
        "      <th>37</th>\n"
        "      <th>38</th>\n"
        "      <th>39</th>\n"
        "      <th>...</th>\n"
        "      <th>36028797018963928</th>\n"
        "      <th>36028797018963929</th>\n"
        "      <th>36028797018963930</th>\n"
        "      <th>36028797018963931</th>\n"
        "      <th>36028797018963932</th>\n"
        "      <th>36028797018963933</th>\n"
        "      <th>36028797018963934</th>\n"
        "      <th>36028797018963935</th>\n"
        "      <th>36028797018963936</th>\n"
        "      <th>36028797018963937</th>\n"
        "      <th>36028797018963938</th>\n"
        "      <th>36028797018963939</th>\n"
        "      <th>36028797018963940</th>\n"
        "      <th>36028797018963941</th>\n"
        "      <th>36028797018963942</th>\n"
        "      <th>36028797018963943</th>\n"
        "      <th>36028797018963944</th>\n"
        "      <th>36028797018963945</th>\n"
        "      <th>36028797018963946</th>\n"
        "      <th>36028797018963947</th>\n"
        "      <th>36028797018963948</th>\n"
        "      <th>36028797018963949</th>\n"
        "      <th>36028797018963950</th>\n"
        "      <th>36028797018963951</th>\n"
        "      <th>36028797018963952</th>\n"
        "      <th>36028797018963953</th>\n"
        "      <th>36028797018963954</th>\n"
        "      <th>36028797018963955</th>\n"
        "      <th>36028797018963956</th>\n"
        "      <th>36028797018963957</th>\n"
        "      <th>36028797018963958</th>\n"
        "      <th>36028797018963959</th>\n"
        "      <th>36028797018963960</th>\n"
        "      <th>36028797018963961</th>\n"
        "      <th>36028797018963962</th>\n"
        "      <th>36028797018963963</th>\n"
        "      <th>36028797018963964</th>\n"
        "      <th>36028797018963965</th>\n"
        "      <th>36028797018963966</th>\n"
        "      <th>36028797018963967</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        "      <td>2</td>\n"
        + "      <td></td>\n" * 39
        + "      <td>...</td>\n"
        + "      <td></td>\n" * 40
        + "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>Do <code>expr.new()</code> or <code>other << expr</code> to calculate the expression.</em></div>"
    )
    html_printer(BIG.ewise_add(small), "BIG.ewise_add(small)")
    assert repr_html(BIG.ewise_add(small)) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.VectorExpression:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>v<sub>0</sub>.ewise_add(v<sub>1</sub>, op=monoid.plus[INT64])</pre></td>\n'
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>36028797018963968</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v<sub>0</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n" + "    <td>36028797018963968</td>\n" * 2 + "    <td>INT64</td>\n"
        "    <td>full (iso)</td>\n"
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
        "      <th>10</th>\n"
        "      <th>11</th>\n"
        "      <th>12</th>\n"
        "      <th>13</th>\n"
        "      <th>14</th>\n"
        "      <th>15</th>\n"
        "      <th>16</th>\n"
        "      <th>17</th>\n"
        "      <th>18</th>\n"
        "      <th>19</th>\n"
        "      <th>20</th>\n"
        "      <th>21</th>\n"
        "      <th>22</th>\n"
        "      <th>23</th>\n"
        "      <th>24</th>\n"
        "      <th>25</th>\n"
        "      <th>26</th>\n"
        "      <th>27</th>\n"
        "      <th>28</th>\n"
        "      <th>29</th>\n"
        "      <th>30</th>\n"
        "      <th>31</th>\n"
        "      <th>32</th>\n"
        "      <th>33</th>\n"
        "      <th>34</th>\n"
        "      <th>35</th>\n"
        "      <th>36</th>\n"
        "      <th>37</th>\n"
        "      <th>38</th>\n"
        "      <th>39</th>\n"
        "      <th>...</th>\n"
        "      <th>36028797018963928</th>\n"
        "      <th>36028797018963929</th>\n"
        "      <th>36028797018963930</th>\n"
        "      <th>36028797018963931</th>\n"
        "      <th>36028797018963932</th>\n"
        "      <th>36028797018963933</th>\n"
        "      <th>36028797018963934</th>\n"
        "      <th>36028797018963935</th>\n"
        "      <th>36028797018963936</th>\n"
        "      <th>36028797018963937</th>\n"
        "      <th>36028797018963938</th>\n"
        "      <th>36028797018963939</th>\n"
        "      <th>36028797018963940</th>\n"
        "      <th>36028797018963941</th>\n"
        "      <th>36028797018963942</th>\n"
        "      <th>36028797018963943</th>\n"
        "      <th>36028797018963944</th>\n"
        "      <th>36028797018963945</th>\n"
        "      <th>36028797018963946</th>\n"
        "      <th>36028797018963947</th>\n"
        "      <th>36028797018963948</th>\n"
        "      <th>36028797018963949</th>\n"
        "      <th>36028797018963950</th>\n"
        "      <th>36028797018963951</th>\n"
        "      <th>36028797018963952</th>\n"
        "      <th>36028797018963953</th>\n"
        "      <th>36028797018963954</th>\n"
        "      <th>36028797018963955</th>\n"
        "      <th>36028797018963956</th>\n"
        "      <th>36028797018963957</th>\n"
        "      <th>36028797018963958</th>\n"
        "      <th>36028797018963959</th>\n"
        "      <th>36028797018963960</th>\n"
        "      <th>36028797018963961</th>\n"
        "      <th>36028797018963962</th>\n"
        "      <th>36028797018963963</th>\n"
        "      <th>36028797018963964</th>\n"
        "      <th>36028797018963965</th>\n"
        "      <th>36028797018963966</th>\n"
        "      <th>36028797018963967</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        + "      <td>1</td>\n" * 40
        + "      <td>...</td>\n"
        + "      <td>1</td>\n" * 40
        + "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>1</td>\n"
        "    <td>36028797018963968</td>\n"
        "    <td>INT64</td>\n"
        "    <td>sparse (iso)</td>\n"
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
        "      <th>10</th>\n"
        "      <th>11</th>\n"
        "      <th>12</th>\n"
        "      <th>13</th>\n"
        "      <th>14</th>\n"
        "      <th>15</th>\n"
        "      <th>16</th>\n"
        "      <th>17</th>\n"
        "      <th>18</th>\n"
        "      <th>19</th>\n"
        "      <th>20</th>\n"
        "      <th>21</th>\n"
        "      <th>22</th>\n"
        "      <th>23</th>\n"
        "      <th>24</th>\n"
        "      <th>25</th>\n"
        "      <th>26</th>\n"
        "      <th>27</th>\n"
        "      <th>28</th>\n"
        "      <th>29</th>\n"
        "      <th>30</th>\n"
        "      <th>31</th>\n"
        "      <th>32</th>\n"
        "      <th>33</th>\n"
        "      <th>34</th>\n"
        "      <th>35</th>\n"
        "      <th>36</th>\n"
        "      <th>37</th>\n"
        "      <th>38</th>\n"
        "      <th>39</th>\n"
        "      <th>...</th>\n"
        "      <th>36028797018963928</th>\n"
        "      <th>36028797018963929</th>\n"
        "      <th>36028797018963930</th>\n"
        "      <th>36028797018963931</th>\n"
        "      <th>36028797018963932</th>\n"
        "      <th>36028797018963933</th>\n"
        "      <th>36028797018963934</th>\n"
        "      <th>36028797018963935</th>\n"
        "      <th>36028797018963936</th>\n"
        "      <th>36028797018963937</th>\n"
        "      <th>36028797018963938</th>\n"
        "      <th>36028797018963939</th>\n"
        "      <th>36028797018963940</th>\n"
        "      <th>36028797018963941</th>\n"
        "      <th>36028797018963942</th>\n"
        "      <th>36028797018963943</th>\n"
        "      <th>36028797018963944</th>\n"
        "      <th>36028797018963945</th>\n"
        "      <th>36028797018963946</th>\n"
        "      <th>36028797018963947</th>\n"
        "      <th>36028797018963948</th>\n"
        "      <th>36028797018963949</th>\n"
        "      <th>36028797018963950</th>\n"
        "      <th>36028797018963951</th>\n"
        "      <th>36028797018963952</th>\n"
        "      <th>36028797018963953</th>\n"
        "      <th>36028797018963954</th>\n"
        "      <th>36028797018963955</th>\n"
        "      <th>36028797018963956</th>\n"
        "      <th>36028797018963957</th>\n"
        "      <th>36028797018963958</th>\n"
        "      <th>36028797018963959</th>\n"
        "      <th>36028797018963960</th>\n"
        "      <th>36028797018963961</th>\n"
        "      <th>36028797018963962</th>\n"
        "      <th>36028797018963963</th>\n"
        "      <th>36028797018963964</th>\n"
        "      <th>36028797018963965</th>\n"
        "      <th>36028797018963966</th>\n"
        "      <th>36028797018963967</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        "      <td>2</td>\n"
        + "      <td></td>\n" * 39
        + "      <td>...</td>\n"
        + "      <td></td>\n" * 40
        + "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div><hr><b>Result is too large to compute!</b></blockquote></details><em>Do <code>expr.new()</code> or <code>other << expr</code> to calculate the expression.</em></div>"
    )
    BIG_bool = BIG.dup(dtype=bool)
    small_bool = small.dup(dtype=bool)
    small_bool[0] = False
    html_printer(BIG_bool | small_bool, "BIG_bool | small_bool")
    assert repr_html(BIG_bool | small_bool) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.VectorEwiseAddExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>v<sub>6</sub> | v<sub>7</sub></pre></td>\n'
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>left_dtype</pre></td>\n"
        "    <td><pre>right_dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>36028797018963968</td>\n" + "    <td>BOOL</td>\n" * 2 + "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v<sub>6</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n" + "    <td>36028797018963968</td>\n" * 2 + "    <td>BOOL</td>\n"
        "    <td>full (iso)</td>\n"
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
        "      <th>10</th>\n"
        "      <th>11</th>\n"
        "      <th>12</th>\n"
        "      <th>13</th>\n"
        "      <th>14</th>\n"
        "      <th>15</th>\n"
        "      <th>16</th>\n"
        "      <th>17</th>\n"
        "      <th>18</th>\n"
        "      <th>19</th>\n"
        "      <th>20</th>\n"
        "      <th>21</th>\n"
        "      <th>22</th>\n"
        "      <th>23</th>\n"
        "      <th>24</th>\n"
        "      <th>25</th>\n"
        "      <th>26</th>\n"
        "      <th>27</th>\n"
        "      <th>28</th>\n"
        "      <th>29</th>\n"
        "      <th>30</th>\n"
        "      <th>31</th>\n"
        "      <th>32</th>\n"
        "      <th>33</th>\n"
        "      <th>34</th>\n"
        "      <th>35</th>\n"
        "      <th>36</th>\n"
        "      <th>37</th>\n"
        "      <th>38</th>\n"
        "      <th>39</th>\n"
        "      <th>...</th>\n"
        "      <th>36028797018963928</th>\n"
        "      <th>36028797018963929</th>\n"
        "      <th>36028797018963930</th>\n"
        "      <th>36028797018963931</th>\n"
        "      <th>36028797018963932</th>\n"
        "      <th>36028797018963933</th>\n"
        "      <th>36028797018963934</th>\n"
        "      <th>36028797018963935</th>\n"
        "      <th>36028797018963936</th>\n"
        "      <th>36028797018963937</th>\n"
        "      <th>36028797018963938</th>\n"
        "      <th>36028797018963939</th>\n"
        "      <th>36028797018963940</th>\n"
        "      <th>36028797018963941</th>\n"
        "      <th>36028797018963942</th>\n"
        "      <th>36028797018963943</th>\n"
        "      <th>36028797018963944</th>\n"
        "      <th>36028797018963945</th>\n"
        "      <th>36028797018963946</th>\n"
        "      <th>36028797018963947</th>\n"
        "      <th>36028797018963948</th>\n"
        "      <th>36028797018963949</th>\n"
        "      <th>36028797018963950</th>\n"
        "      <th>36028797018963951</th>\n"
        "      <th>36028797018963952</th>\n"
        "      <th>36028797018963953</th>\n"
        "      <th>36028797018963954</th>\n"
        "      <th>36028797018963955</th>\n"
        "      <th>36028797018963956</th>\n"
        "      <th>36028797018963957</th>\n"
        "      <th>36028797018963958</th>\n"
        "      <th>36028797018963959</th>\n"
        "      <th>36028797018963960</th>\n"
        "      <th>36028797018963961</th>\n"
        "      <th>36028797018963962</th>\n"
        "      <th>36028797018963963</th>\n"
        "      <th>36028797018963964</th>\n"
        "      <th>36028797018963965</th>\n"
        "      <th>36028797018963966</th>\n"
        "      <th>36028797018963967</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        + "      <td>True</td>\n" * 40
        + "      <td>...</td>\n"
        + "      <td>True</td>\n" * 40
        + "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v<sub>7</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>1</td>\n"
        "    <td>36028797018963968</td>\n"
        "    <td>BOOL</td>\n"
        "    <td>sparse</td>\n"
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
        "      <th>10</th>\n"
        "      <th>11</th>\n"
        "      <th>12</th>\n"
        "      <th>13</th>\n"
        "      <th>14</th>\n"
        "      <th>15</th>\n"
        "      <th>16</th>\n"
        "      <th>17</th>\n"
        "      <th>18</th>\n"
        "      <th>19</th>\n"
        "      <th>20</th>\n"
        "      <th>21</th>\n"
        "      <th>22</th>\n"
        "      <th>23</th>\n"
        "      <th>24</th>\n"
        "      <th>25</th>\n"
        "      <th>26</th>\n"
        "      <th>27</th>\n"
        "      <th>28</th>\n"
        "      <th>29</th>\n"
        "      <th>30</th>\n"
        "      <th>31</th>\n"
        "      <th>32</th>\n"
        "      <th>33</th>\n"
        "      <th>34</th>\n"
        "      <th>35</th>\n"
        "      <th>36</th>\n"
        "      <th>37</th>\n"
        "      <th>38</th>\n"
        "      <th>39</th>\n"
        "      <th>...</th>\n"
        "      <th>36028797018963928</th>\n"
        "      <th>36028797018963929</th>\n"
        "      <th>36028797018963930</th>\n"
        "      <th>36028797018963931</th>\n"
        "      <th>36028797018963932</th>\n"
        "      <th>36028797018963933</th>\n"
        "      <th>36028797018963934</th>\n"
        "      <th>36028797018963935</th>\n"
        "      <th>36028797018963936</th>\n"
        "      <th>36028797018963937</th>\n"
        "      <th>36028797018963938</th>\n"
        "      <th>36028797018963939</th>\n"
        "      <th>36028797018963940</th>\n"
        "      <th>36028797018963941</th>\n"
        "      <th>36028797018963942</th>\n"
        "      <th>36028797018963943</th>\n"
        "      <th>36028797018963944</th>\n"
        "      <th>36028797018963945</th>\n"
        "      <th>36028797018963946</th>\n"
        "      <th>36028797018963947</th>\n"
        "      <th>36028797018963948</th>\n"
        "      <th>36028797018963949</th>\n"
        "      <th>36028797018963950</th>\n"
        "      <th>36028797018963951</th>\n"
        "      <th>36028797018963952</th>\n"
        "      <th>36028797018963953</th>\n"
        "      <th>36028797018963954</th>\n"
        "      <th>36028797018963955</th>\n"
        "      <th>36028797018963956</th>\n"
        "      <th>36028797018963957</th>\n"
        "      <th>36028797018963958</th>\n"
        "      <th>36028797018963959</th>\n"
        "      <th>36028797018963960</th>\n"
        "      <th>36028797018963961</th>\n"
        "      <th>36028797018963962</th>\n"
        "      <th>36028797018963963</th>\n"
        "      <th>36028797018963964</th>\n"
        "      <th>36028797018963965</th>\n"
        "      <th>36028797018963966</th>\n"
        "      <th>36028797018963967</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        "      <td>False</td>\n"
        + "      <td></td>\n" * 39
        + "      <td>...</td>\n"
        + "      <td></td>\n" * 40
        + "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div><hr><b>Result is too large to compute!</b></blockquote></details><em>Do <code>op(expr)</code> to create a <tt>VectorExpression</tt> for <tt>ewise_add</tt>.<br>For example: <code>plus(v<sub>6</sub> | v<sub>7</sub>)</code></em></div>"
    )
    C = A.dup(dtype=bool)
    html_printer(C & C, "C & C")
    assert repr_html(C & C) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.MatrixEwiseMultExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>M<sub>2</sub> & M<sub>2</sub></pre></td>\n'
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>right_dtype</pre></td>\n"
        "    <td><pre>left_dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n" + "    <td>BOOL</td>\n" * 2 + "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>M<sub>2</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>BOOL</td>\n"
        "    <td>bitmapr</td>\n"
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
        "      <td>False</td>\n"
        "      <td></td>\n"
        "      <td>True</td>\n"
        "      <td></td>\n"
        "      <td>True</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div><div>"
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>M<sub>2</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>BOOL</td>\n"
        "    <td>bitmapr</td>\n"
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
        "      <td>False</td>\n"
        "      <td></td>\n"
        "      <td>True</td>\n"
        "      <td></td>\n"
        "      <td>True</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div><hr><div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>Result</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>BOOL</td>\n"
        "    <td>bitmapr</td>\n"
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
        "      <td>False</td>\n"
        "      <td></td>\n"
        "      <td>True</td>\n"
        "      <td></td>\n"
        "      <td>True</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>Do <code>op(expr)</code> to create a <tt>MatrixExpression</tt> for <tt>ewise_mult</tt>.<br>For example: <code>times(M<sub>2</sub> & M<sub>2</sub>)</code></em></div>"
    )


@pytest.mark.skipif("not pd")
def test_display_nan():
    v = Vector.from_values([0, 1], [1.0, np.nan], size=3, name="v")
    repr_printer(v, "v")
    assert repr(v) == (
        '"v"        nvals  size  dtype  format\n'
        "gb.Vector      2     3   FP64  bitmap\n"
        "-------------------------------------\n"
        "index    0    1 2\n"
        "value  1.0  nan  "
    )
    html_printer(v, "v")
    assert repr_html(v) == (
        "<div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>2</td>\n"
        "    <td>3</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        "      <td>1.0</td>\n"
        "      <td>nan</td>\n"
        "      <td></td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )
    A = Matrix.from_values([0, 0], [0, 1], [1.0, np.nan], ncols=3, nrows=2, name="A")
    repr_printer(A, "A")
    assert repr(A) == (
        '"A"        nvals  nrows  ncols  dtype   format\n'
        "gb.Matrix      2      2      3   FP64  bitmapr\n"
        "----------------------------------------------\n"
        "     0    1 2\n"
        "0  1.0  nan  \n"
        "1            "
    )
    html_printer(A, "A")
    assert repr_html(A) == (
        "<div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>A</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n" + "    <td>2</td>\n" * 2 + "    <td>3</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>1.0</td>\n"
        "      <td>nan</td>\n"
        "      <td></td>\n"
        "    </tr>\n"
        "    <tr>\n"
        "      <th>1</th>\n" + "      <td></td>\n" * 3 + "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div>"
    )


@pytest.mark.skipif("not pd")
def test_large_iso():
    A = Matrix(int, nrows=2**60, ncols=2**60)
    A[:, :] << 1
    repr_printer(A, "A")
    assert repr(A) == (
        '"M_0"                    nvals                nrows                ncols  dtype       format\n'
        "gb.Matrix  9223372036854775807  1152921504606846976  1152921504606846976  INT64  fullr (iso)\n"
        "--------------------------------------------------------------------------------------------\n"
        "                    0                    ... 1152921504606846975\n"
        "0                                     1  ...                   1\n"
        "1                                     1  ...                   1\n"
        "2                                     1  ...                   1\n"
        "3                                     1  ...                   1\n"
        "4                                     1  ...                   1\n"
        "...                                 ...  ...                 ...\n"
        "1152921504606846971                   1  ...                   1\n"
        "1152921504606846972                   1  ...                   1\n"
        "1152921504606846973                   1  ...                   1\n"
        "1152921504606846974                   1  ...                   1\n"
        "1152921504606846975                   1  ...                   1"
    )
    repr_printer(A.T, "A.T")
    assert repr(A.T) == (
        '"M_0.T"                            nvals                nrows                ncols  dtype       format\n'
        "gb.TransposedMatrix  9223372036854775807  1152921504606846976  1152921504606846976  INT64  fullc (iso)\n"
        "------------------------------------------------------------------------------------------------------\n"
        "                    0                    ... 1152921504606846975\n"
        "0                                     1  ...                   1\n"
        "1                                     1  ...                   1\n"
        "2                                     1  ...                   1\n"
        "3                                     1  ...                   1\n"
        "4                                     1  ...                   1\n"
        "...                                 ...  ...                 ...\n"
        "1152921504606846971                   1  ...                   1\n"
        "1152921504606846972                   1  ...                   1\n"
        "1152921504606846973                   1  ...                   1\n"
        "1152921504606846974                   1  ...                   1\n"
        "1152921504606846975                   1  ...                   1"
    )
    repr_printer(A.S, "A.S")
    assert repr(A.S) == (
        '"M_0.S"                       nvals                nrows                ncols  dtype       format\n'
        "StructuralMask\n"
        "of gb.Matrix    9223372036854775807  1152921504606846976  1152921504606846976  INT64  fullr (iso)\n"
        "-------------------------------------------------------------------------------------------------\n"
        "                    0                    ... 1152921504606846975\n"
        "0                                     1  ...                   1\n"
        "1                                     1  ...                   1\n"
        "2                                     1  ...                   1\n"
        "3                                     1  ...                   1\n"
        "4                                     1  ...                   1\n"
        "...                                 ...  ...                 ...\n"
        "1152921504606846971                   1  ...                   1\n"
        "1152921504606846972                   1  ...                   1\n"
        "1152921504606846973                   1  ...                   1\n"
        "1152921504606846974                   1  ...                   1\n"
        "1152921504606846975                   1  ...                   1"
    )
    v = Vector(int, size=2**60)
    v[:] = 1
    repr_printer(v, "v")
    assert repr(v) == (
        '"v_0"                    nvals                 size  dtype      format\n'
        "gb.Vector  1152921504606846976  1152921504606846976  INT64  full (iso)\n"
        "----------------------------------------------------------------------\n"
        "index 0                    ... 1152921504606846975\n"
        "value                   1  ...                   1"
    )
    repr_printer(v.S, "v.S")
    assert repr(v.S) == (
        '"v_0.S"                       nvals                 size  dtype      format\n'
        "StructuralMask\n"
        "of gb.Vector    1152921504606846976  1152921504606846976  INT64  full (iso)\n"
        "---------------------------------------------------------------------------\n"
        "index 0                    ... 1152921504606846975\n"
        "value                   1  ...                   1"
    )


def test_index_expr_vector(v):
    repr_printer(v[0], "v[0]")
    assert repr(v[0]) == (
        "gb.ScalarIndexExpr  dtype\n"
        "v[0]                 FP64\n"
        "\n"
        "This expression may be used to extract or assign a Scalar.\n"
        "Example extract: v[0].new()\n"
        "Example assign: v[0] << s"
    )
    repr_printer(v[[0, 1]], "v[[0, 1]]")
    assert repr(v[[0, 1]]) == (
        "gb.VectorIndexExpr  size  dtype\n"
        "v[[0, 1]]              2   FP64\n"
        "\n"
        "This expression may be used to extract or assign a Vector.\n"
        "Example extract: v[[0, 1]].new()\n"
        "Example assign: v[[0, 1]] << v"
    )
    repr_printer(v[1:3], "v[1:3]")
    assert repr(v[1:3]) == (
        "gb.VectorIndexExpr  size  dtype\n"
        "v[1:3]                 2   FP64\n"
        "\n"
        "This expression may be used to extract or assign a Vector.\n"
        "Example extract: v[1:3].new()\n"
        "Example assign: v[1:3] << v"
    )
    repr_printer(v[::2], "v[::2]")
    assert repr(v[::2]) == (
        "gb.VectorIndexExpr  size  dtype\n"
        "v[::2]                 3   FP64\n"
        "\n"
        "This expression may be used to extract or assign a Vector.\n"
        "Example extract: v[::2].new()\n"
        "Example assign: v[::2] << v"
    )
    repr_printer(v[[0] * 50], "v[[0] * 50]")
    assert repr(v[[0] * 50]) == (
        "gb.VectorIndexExpr  size  dtype\n"
        "v[[0, 0, 0, ...]]     50   FP64\n"
        "\n"
        "This expression may be used to extract or assign a Vector.\n"
        "Example extract: v[[0, 0, 0, ...]].new()\n"
        "Example assign: v[[0, 0, 0, ...]] << v"
    )


def test_index_expr_vector_html(v):
    html_printer(v[0], "v[0]")
    assert repr_html(v[0]) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.ScalarIndexExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>v[0]</pre></td>\n'
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>This expression may be used to extract or assign a <tt>Scalar</tt>.<br>Example extract: <code>v[0].new()</code><br>Example assign: <code>v[0] << s</code></em></div>"
    )
    html_printer(v[[0, 1]], "v[[0, 1]]")
    assert repr_html(v[[0, 1]]) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.VectorIndexExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>v[[0, 1]]</pre></td>\n'
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>2</td>\n"
        "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>This expression may be used to extract or assign a <tt>Vector</tt>.<br>Example extract: <code>v[[0, 1]].new()</code><br>Example assign: <code>v[[0, 1]] << v</code></em></div>"
    )


def test_index_expr_matrix(A):
    repr_printer(A[0, 1], "A[0, 1]")
    assert repr(A[0, 1]) == (
        "gb.ScalarIndexExpr  dtype\n"
        "A_1[0, 1]           INT64\n"
        "\n"
        "This expression may be used to extract or assign a Scalar.\n"
        "Example extract: A_1[0, 1].new()\n"
        "Example assign: A_1[0, 1] << s"
    )
    repr_printer(A[:2, [1, 2]], "A[:2, [1, 2]]")
    assert repr(A[:2, [1, 2]]) == (
        "gb.MatrixIndexExpr  nrows  ncols  dtype\n"
        "A_1[:, [1, 2]]          1      2  INT64\n"
        "\n"
        "This expression may be used to extract or assign a Matrix.\n"
        "Example extract: A_1[:, [1, 2]].new()\n"
        "Example assign: A_1[:, [1, 2]] << M"
    )


def test_index_expr_matrix_html(A):
    html_printer(A[0, 1], "A[0, 1]")
    assert repr_html(A[0, 1]) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.ScalarIndexExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>A<sub>1</sub>[0, 1]</pre></td>\n'
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "</div></details></div></blockquote></details><em>This expression may be used to extract or assign a <tt>Scalar</tt>.<br>Example extract: <code>A<sub>1</sub>[0, 1].new()</code><br>Example assign: <code>A<sub>1</sub>[0, 1] << s</code></em></div>"
    )
    html_printer(A[:2, [1, 2]], "A[:2, [1, 2]]")
    assert repr_html(A[:2, [1, 2]]) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.MatrixIndexExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>A<sub>1</sub>[:, [1, 2]]</pre></td>\n'
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>1</td>\n"
        "    <td>2</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>A<sub>1</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Matrix</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>nrows</pre></td>\n"
        "    <td><pre>ncols</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>1</td>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "    <td>bitmapr</td>\n"
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
        "</div></details></div></blockquote></details><em>This expression may be used to extract or assign a <tt>Matrix</tt>.<br>Example extract: <code>A<sub>1</sub>[:, [1, 2]].new()</code><br>Example assign: <code>A<sub>1</sub>[:, [1, 2]] << M</code></em></div>"
    )


def test_scalar_as_vector():
    s = Scalar.from_value(5, is_cscalar=False)  # pragma: is_grbscalar
    v = s._as_vector()
    repr_printer(v, "v")
    assert repr(v) == (
        '"(GrB_Vector)s_0"  nvals  size  dtype      format\n'
        "gb.Vector              1     1  INT64  full (iso)\n"
        "-------------------------------------------------\n"
        "index  0\n"
        "value  5"
    )
    expr = v.reduce()
    repr_printer(expr, "expr")
    assert repr(expr) == (
        "gb.ScalarExpression                         dtype\n"
        "(GrB_Vector)s_0.reduce(monoid.plus[INT64])  INT64\n"
        "\n"
        "Do expr.new() or other << expr to calculate the expression."
    )
    html_printer(v, "v")
    assert repr_html(v) == (
        '<div class="gb-scalar"><tt>s<sub>0</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Scalar</pre></td>\n'
        "    <td><pre>value</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</div>"
    )
    html_printer(expr, "expr")
    assert repr_html(expr) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.ScalarExpression:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>s<sub>0</sub>.reduce(monoid.plus[INT64])</pre></td>\n'
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div class="gb-scalar"><tt>s<sub>0</sub></tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Scalar</pre></td>\n'
        "    <td><pre>value</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>5</td>\n"
        "    <td>INT64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        "</div></blockquote></details><em>Do <code>expr.new()</code> or <code>other << expr</code> to calculate the expression.</em></div>"
    )


@autocompute
def test_index_expr_autocompute(v):
    html_printer(v[[0, 1]], "v[[0, 1]]")
    assert repr_html(v[[0, 1]]) == (
        '<div><details class="gb-expr-details"><summary class="gb-expr-summary"><b><tt>gb.VectorIndexExpr:</tt></b><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>v[[0, 1]]</pre></td>\n'
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>2</td>\n"
        "    <td>FP64</td>\n"
        "  </tr>\n"
        "</table>\n"
        "</div>\n"
        '</summary><blockquote class="gb-expr-blockquote"><div>'
        f"{CSS_STYLE}"
        '<details class="gb-arg-details"><summary class="gb-arg-summary"><tt>v</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>3</td>\n"
        "    <td>5</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "      <td>1.1</td>\n"
        "      <td></td>\n"
        "      <td>2.2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div><hr><div>"
        f"{CSS_STYLE}"
        '<details open class="gb-arg-details"><summary class="gb-arg-summary"><tt>Result</tt><div>\n'
        '<table class="gb-info-table">\n'
        "  <tr>\n"
        '    <td rowspan="2" class="gb-info-name-cell"><pre>gb.Vector</pre></td>\n'
        "    <td><pre>nvals</pre></td>\n"
        "    <td><pre>size</pre></td>\n"
        "    <td><pre>dtype</pre></td>\n"
        "    <td><pre>format</pre></td>\n"
        "  </tr>\n"
        "  <tr>\n"
        "    <td>1</td>\n"
        "    <td>2</td>\n"
        "    <td>FP64</td>\n"
        "    <td>bitmap</td>\n"
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
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th></th>\n"
        "      <td>0.0</td>\n"
        "      <td></td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>\n"
        "</div></details></div></blockquote></details><em>This expression may be used to extract or assign a <tt>Vector</tt>.<br>Example extract: <code>v[[0, 1]].new()</code><br>Example assign: <code>v[[0, 1]] << v</code></em></div>"
    )


def test_udt():
    record_dtype = np.dtype([("x", np.bool_), ("y", np.int64)], align=True)
    udt = dtypes.register_anonymous(record_dtype, "record_dtype")
    v = Vector(udt, size=2)
    v[:] = (False, 2)
    v[1] = (True, 3)
    repr_printer(v, "v")
    assert repr(v) == (
        '"v_0"      nvals  size         dtype  format\n'
        "gb.Vector      2     2  record_dtype    full\n"
        "--------------------------------------------\n"
        "index           0          1\n"
        "value  (False, 2)  (True, 3)"
    )
    A = Matrix(udt, nrows=1, ncols=3)
    A[:, :2] = 0
    repr_printer(A, "A")
    assert repr(A) == (
        '"M_0"      nvals  nrows  ncols         dtype         format\n'
        "gb.Matrix      2      1      3  record_dtype  bitmapr (iso)\n"
        "-----------------------------------------------------------\n"
        "            0           1 2\n"
        "0  (False, 0)  (False, 0)  "
    )
    np_dtype = np.dtype("(3,)uint16")
    udt2 = dtypes.register_anonymous(np_dtype, "has_subdtype")
    v = Vector(udt2, size=2)
    v[:] = (1, 2, 3)
    v[1] = (3, 2, 1)
    repr_printer(v, "v")
    assert repr(v) == (
        '"v_1"      nvals  size         dtype  format\n'
        "gb.Vector      2     2  has_subdtype    full\n"
        "--------------------------------------------\n"
        "index          0          1\n"
        "value  [1, 2, 3]  [3, 2, 1]"
    )
    A = Matrix(udt2, nrows=1, ncols=3)
    A[:, :2] = 1
    repr_printer(A, "A")
    assert repr(A) == (
        '"M_1"      nvals  nrows  ncols         dtype         format\n'
        "gb.Matrix      2      1      3  has_subdtype  bitmapr (iso)\n"
        "-----------------------------------------------------------\n"
        "           0          1 2\n"
        "0  [1, 1, 1]  [1, 1, 1]  "
    )


def test_empty():
    v = Vector(int, 0)
    repr_printer(v, "v")
    assert repr(v) == (
        '"v_0"      nvals  size  dtype  format\n'
        "gb.Vector      0     0  INT64  sparse\n"
        "-------------------------------------"
    )
    A = Matrix(int, 0, 0)
    repr_printer(A, "A")
    assert repr(A) == (
        '"M_0"      nvals  nrows  ncols  dtype  format\n'
        "gb.Matrix      0      0      0  INT64     csr\n"
        "---------------------------------------------"
    )
    A = Matrix(int, 0, 5)
    repr_printer(A, "A")
    assert repr(A) == (
        '"M_1"      nvals  nrows  ncols  dtype  format\n'
        "gb.Matrix      0      0      5  INT64     csr\n"
        "---------------------------------------------"
    )
