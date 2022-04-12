import pytest

import graphblas as gb
from graphblas.exceptions import OutOfMemory
from graphblas.formatting import CSS_STYLE


@pytest.fixture
def switch():
    return gb.Scalar.from_value(5)._is_cscalar


def test_recorder():
    A = gb.Matrix.from_values([0, 1], [1, 1], [1, 2], name="A")
    B = gb.Matrix.from_values([0, 1], [0, 1], [3, 4], name="B")
    with gb.Recorder() as rec:
        assert rec.is_recording
        C = A.mxm(B).new(name="C")
    assert not rec.is_recording
    with rec:
        assert rec.is_recording
        rec.start()  # no-op
        D = A.mxm(B.T, gb.semiring.min_plus).new(name="D")
        C(D.S) << A.T.ewise_mult(B)
    A.mxm(B).new(name="E")  # not recorded
    rec.stop()  # no-op

    assert len(rec.data) == 5
    assert list(rec) == [
        "GrB_Matrix_new(&C, GrB_INT64, 2, 2);",
        "GrB_mxm(C, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_INT64, A, B, NULL);",
        "GrB_Matrix_new(&D, GrB_INT64, 2, 2);",
        "GrB_mxm(D, NULL, NULL, GrB_MIN_PLUS_SEMIRING_INT64, A, B, GrB_DESC_T1);",
        "GrB_Matrix_eWiseMult_BinaryOp(C, D, NULL, GrB_TIMES_INT64, A, B, GrB_DESC_ST0);",
    ]
    rec.clear()
    assert list(rec) == []


def test_record_novalue(switch):
    A = gb.Matrix(int, 3, 3, name="A")
    rec = gb.Recorder(start=True)
    A[0, 0].new(name="c")
    if switch:
        assert rec.data == ["GrB_Matrix_extractElement_INT64(&c, A, 0, 0);"]
    else:
        assert rec.data == [
            "GrB_Scalar_new(&c, GrB_INT64);",
            "GrB_Matrix_extractElement_Scalar(c, A, 0, 0);",
        ]
    rec.stop()


def test_record_scalars(switch):
    A = gb.Matrix(int, 3, 3, name="A")
    with gb.Recorder() as rec:
        A[0, 0] = 5
        A.apply(gb.binary.lt, right=10).new(name="B")
    if switch:
        assert list(rec) == [
            # assign
            "GrB_Scalar_new(&s_temp, GrB_INT64);",
            "GrB_Scalar_setElement_INT64(s_temp, 5);",
            "GrB_Matrix_setElement_Scalar(A, s_temp, 0, 0);",
            # apply
            "GrB_Scalar_new(&s_temp, GrB_INT64);",
            "GrB_Scalar_setElement_INT64(s_temp, 10);",
            "GrB_Matrix_new(&B, GrB_BOOL, 3, 3);",
            "GrB_Matrix_apply_BinaryOp2nd_Scalar(B, NULL, NULL, GrB_LT_INT64, A, s_temp, NULL);",
        ]
    else:
        assert list(rec) == [
            "GrB_Matrix_setElement_INT64(A, 5, 0, 0);",
            "GrB_Matrix_new(&B, GrB_BOOL, 3, 3);",
            "GrB_Matrix_apply_BinaryOp2nd_INT64(B, NULL, NULL, GrB_LT_INT64, A, 10, NULL);",
        ]


def test_record_repr(switch):
    A = gb.Matrix(int, 3, 3, name="A")
    rec = gb.Recorder(start=True)
    A[0, 0].new(name="c0")
    if switch:
        assert repr(rec) == (
            "gb.Recorder (recording)\n"
            "-----------------------\n"
            "  GrB_Matrix_extractElement_INT64(&c0, A, 0, 0);"
        )
    else:
        assert repr(rec) == (
            "gb.Recorder (recording)\n"
            "-----------------------\n"
            "  GrB_Scalar_new(&c0, GrB_INT64);\n"
            "  GrB_Matrix_extractElement_Scalar(c0, A, 0, 0);"
        )
    rec.stop()
    if switch:
        assert repr(rec) == (
            "gb.Recorder (not recording)\n"
            "---------------------------\n"
            "  GrB_Matrix_extractElement_INT64(&c0, A, 0, 0);"
        )
    else:
        assert repr(rec) == (
            "gb.Recorder (not recording)\n"
            "---------------------------\n"
            "  GrB_Scalar_new(&c0, GrB_INT64);\n"
            "  GrB_Matrix_extractElement_Scalar(c0, A, 0, 0);"
        )
    rec.start()
    rec.max_rows = 10
    for i in range(1, 20):
        A[0, 0].new(name=f"c{i}")
    if switch:
        assert repr(rec) == (
            "gb.Recorder (recording)\n"
            "-----------------------\n"
            "  GrB_Matrix_extractElement_INT64(&c0, A, 0, 0);\n"
            "  GrB_Matrix_extractElement_INT64(&c1, A, 0, 0);\n"
            "  GrB_Matrix_extractElement_INT64(&c2, A, 0, 0);\n"
            "  GrB_Matrix_extractElement_INT64(&c3, A, 0, 0);\n"
            "  GrB_Matrix_extractElement_INT64(&c4, A, 0, 0);\n"
            "\n"
            "  ... (10 rows not shown)\n"
            "\n"
            "  GrB_Matrix_extractElement_INT64(&c15, A, 0, 0);\n"
            "  GrB_Matrix_extractElement_INT64(&c16, A, 0, 0);\n"
            "  GrB_Matrix_extractElement_INT64(&c17, A, 0, 0);\n"
            "  GrB_Matrix_extractElement_INT64(&c18, A, 0, 0);\n"
            "  GrB_Matrix_extractElement_INT64(&c19, A, 0, 0);"
        )
    else:
        assert repr(rec) == (
            "gb.Recorder (recording)\n"
            "-----------------------\n"
            "  GrB_Scalar_new(&c0, GrB_INT64);\n"
            "  GrB_Matrix_extractElement_Scalar(c0, A, 0, 0);\n"
            "  GrB_Scalar_new(&c1, GrB_INT64);\n"
            "  GrB_Matrix_extractElement_Scalar(c1, A, 0, 0);\n"
            "  GrB_Scalar_new(&c2, GrB_INT64);\n"
            "\n"
            "  ... (30 rows not shown)\n"
            "\n"
            "  GrB_Matrix_extractElement_Scalar(c17, A, 0, 0);\n"
            "  GrB_Scalar_new(&c18, GrB_INT64);\n"
            "  GrB_Matrix_extractElement_Scalar(c18, A, 0, 0);\n"
            "  GrB_Scalar_new(&c19, GrB_INT64);\n"
            "  GrB_Matrix_extractElement_Scalar(c19, A, 0, 0);"
        )
    rec.stop()


def test_record_repr_markdown(switch):
    A = gb.Matrix(int, 3, 3, name="A")
    rec = gb.Recorder()
    rec.start()
    A[0, 0].new(name="c")
    if switch:
        text = "  GrB_Matrix_extractElement_INT64(&c, A, 0, 0);\n"
    else:
        text = (
            "  GrB_Scalar_new(&c, GrB_INT64);\n" "  GrB_Matrix_extractElement_Scalar(c, A, 0, 0);\n"
        )
    assert rec._repr_markdown_() == (
        "<div>\n"
        f"{CSS_STYLE}\n"
        '<details open class="gb-arg-details">\n'
        '<summary class="gb-arg-summary">\n'
        '<table class="gb-info-table" style="display: inline-block; vertical-align: middle;">\n'
        "<tr><td>\n"
        "<tt>gb.Recorder</tt>\n"
        '<div style="height: 12px; width: 12px; display: inline-block; vertical-align: middle; '
        'margin-left: 2px; background-color: red; border-radius: 50%;"></div>\n'
        "</td></tr>\n"
        "</table>\n"
        "</summary>\n"
        '<blockquote class="gb-expr-blockquote" style="margin-left: -8px;">\n'
        "\n"
        "```C\n"
        f"{text}"
        "```\n"
        "</blockquote>\n"
        "</details>\n"
        "</div>"
    )
    rec.stop()
    if switch:
        text = "  GrB_Matrix_extractElement_INT64(&c, A, 0, 0);\n"
    else:
        text = (
            "  GrB_Scalar_new(&c, GrB_INT64);\n" "  GrB_Matrix_extractElement_Scalar(c, A, 0, 0);\n"
        )
    assert rec._repr_markdown_() == (
        "<div>\n"
        f"{CSS_STYLE}\n"
        '<details open class="gb-arg-details">\n'
        '<summary class="gb-arg-summary">\n'
        '<table class="gb-info-table" style="display: inline-block; vertical-align: middle;">\n'
        "<tr><td>\n"
        "<tt>gb.Recorder</tt>\n"
        '<div style="height: 12px; width: 12px; display: inline-block; vertical-align: middle; '
        'margin-left: 2px; border-right: 5px solid gray; border-left: 5px solid gray;"></div>\n'
        "</td></tr>\n"
        "</table>\n"
        "</summary>\n"
        '<blockquote class="gb-expr-blockquote" style="margin-left: -8px;">\n'
        "\n"
        "```C\n"
        f"{text}"
        "```\n"
        "</blockquote>\n"
        "</details>\n"
        "</div>"
    )


def test_record_failed_call():
    BIG = gb.Vector(int, size=2**55)
    small = gb.Vector(int, size=2**55)
    BIG[:] = 1
    small[0] = 2
    rec = gb.Recorder()
    try:
        BIG.ewise_add(small).new()
    except OutOfMemory:
        pass
    assert "ERROR: OutOfMemory" in rec.data[-1]


def test_record_inner(switch):
    v = gb.Vector.from_values([0, 1, 2], 1, size=3)
    with gb.Recorder() as rec:
        (v @ v).new(name="s_0")
    if switch:
        assert repr(rec) == (
            "gb.Recorder (not recording)\n"
            "---------------------------\n"
            "  GrB_Vector_new(&v_1, GrB_INT64, 1);\n"
            "  GrB_vxm(v_1, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_INT64, v_0, "
            "(GrB_Matrix)v_0, NULL);\n"
            "  GrB_Vector_extractElement_INT64(&s_temp, v_1, 0);"
        )
    else:
        assert repr(rec) == (
            "gb.Recorder (not recording)\n"
            "---------------------------\n"
            "  GrB_Scalar_new(&s_0, GrB_INT64);\n"
            "  GrB_vxm((GrB_Vector)s_0, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_INT64, v_0, "
            "(GrB_Matrix)v_0, NULL);"
        )
