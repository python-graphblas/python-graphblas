import grblas as gb
from grblas.formatting import CSS_STYLE


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


def test_record_novalue():
    A = gb.Matrix.new(int, 3, 3, name="A")
    rec = gb.Recorder(record=True)
    A[0, 0].new(name="c")
    assert rec.data == ["GrB_Matrix_extractElement_INT64(&c, A, 0, 0);"]
    rec.stop()


def test_record_scalars():
    A = gb.Matrix.new(int, 3, 3, name="A")
    with gb.Recorder() as rec:
        A[0, 0] = 5
        A.apply(gb.binary.lt, right=10).new(name="B")
    assert list(rec) == [
        "GrB_Matrix_setElement_INT64(A, 5, 0, 0);",
        "GrB_Matrix_new(&B, GrB_BOOL, 3, 3);",
        "GrB_Matrix_apply_BinaryOp2nd_INT64(B, NULL, NULL, GrB_LT_INT64, A, 10, NULL);",
    ]


def test_record_repr():
    A = gb.Matrix.new(int, 3, 3, name="A")
    rec = gb.Recorder(record=True)
    A[0, 0].new(name="c0")
    assert repr(rec) == (
        "grblas.Recorder (recording)\n"
        "---------------------------\n"
        "  GrB_Matrix_extractElement_INT64(&c0, A, 0, 0);"
    )
    rec.stop()
    assert repr(rec) == (
        "grblas.Recorder (not recording)\n"
        "-------------------------------\n"
        "  GrB_Matrix_extractElement_INT64(&c0, A, 0, 0);"
    )
    rec.start()
    rec.max_rows = 10
    for i in range(1, 20):
        A[0, 0].new(name=f"c{i}")
    assert repr(rec) == (
        "grblas.Recorder (recording)\n"
        "---------------------------\n"
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
    rec.stop()


def test_record_repr_markdown():
    A = gb.Matrix.new(int, 3, 3, name="A")
    rec = gb.Recorder()
    rec.start()
    A[0, 0].new(name="c")
    assert rec._repr_markdown_() == (
        "<div>\n"
        f"{CSS_STYLE}\n"
        '<details open class="gb-arg-details">\n'
        '<summary class="gb-arg-summary">\n'
        '<table class="gb-info-table" style="display: inline-block; vertical-align: middle;">\n'
        "<tr><td>\n"
        "<tt>grblas.Recorder</tt>\n"
        '<div style="height: 12px; width: 12px; display: inline-block; vertical-align: middle; '
        'margin-left: 2px; background-color: red; border-radius: 50%;"></div>\n'
        "</td></tr>\n"
        "</table>\n"
        "</summary>\n"
        '<blockquote class="gb-expr-blockquote" style="margin-left: -8px;">\n'
        "\n"
        "```C\n"
        "  GrB_Matrix_extractElement_INT64(&c, A, 0, 0);\n"
        "```\n"
        "</blockquote>\n"
        "</details>\n"
        "</div>"
    )
    rec.stop()
    assert rec._repr_markdown_() == (
        "<div>\n"
        f"{CSS_STYLE}\n"
        '<details open class="gb-arg-details">\n'
        '<summary class="gb-arg-summary">\n'
        '<table class="gb-info-table" style="display: inline-block; vertical-align: middle;">\n'
        "<tr><td>\n"
        "<tt>grblas.Recorder</tt>\n"
        '<div style="height: 12px; width: 12px; display: inline-block; vertical-align: middle; '
        'margin-left: 2px; border-right: 5px solid gray; border-left: 5px solid gray;"></div>\n'
        "</td></tr>\n"
        "</table>\n"
        "</summary>\n"
        '<blockquote class="gb-expr-blockquote" style="margin-left: -8px;">\n'
        "\n"
        "```C\n"
        "  GrB_Matrix_extractElement_INT64(&c, A, 0, 0);\n"
        "```\n"
        "</blockquote>\n"
        "</details>\n"
        "</div>"
    )
