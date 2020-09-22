import grblas as gb


def test_recorder():
    A = gb.Matrix.from_values([0, 1], [1, 1], [1, 2], name="A")
    B = gb.Matrix.from_values([0, 1], [0, 1], [3, 4], name="B")
    with gb.Recorder() as rec:
        C = A.mxm(B).new(name="C")
    with rec:
        rec.start()  # no-op
        D = A.mxm(B.T, gb.semiring.min_plus).new(name="D")
        C(D.S) << A.T.ewise_mult(B)
    A.mxm(B).new(name="E")  # not recorded
    rec.stop()  # no-op

    assert len(rec.data) == 5
    assert rec.data == [
        "GrB_Matrix_new(&C, GrB_INT64, 2, 2)",
        "GrB_mxm(C, NULL, NULL, GxB_PLUS_TIMES_INT64, A, B, NULL)",
        "GrB_Matrix_new(&D, GrB_INT64, 2, 2)",
        "GrB_mxm(D, NULL, NULL, GxB_MIN_PLUS_INT64, A, B, GrB_DESC_T1)",
        "GrB_eWiseMult_Matrix_BinaryOp(C, D, NULL, GrB_TIMES_INT64, A, B, GrB_DESC_ST0)",
    ]
