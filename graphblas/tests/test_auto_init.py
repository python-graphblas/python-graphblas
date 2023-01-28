if __name__ == "__main__":
    import pytest

    import graphblas as gb

    with pytest.raises(ValueError, match="Bad backend name"):
        gb.init("bad_name")

    gb.core.ffi
    gb.op
    gb.Matrix
    with pytest.raises(
        gb.exceptions.GraphblasException,
        match="graphblas objects accessed prior to manual initialization",
    ):
        gb.init()
    # hack it to test this edge case
    gb._init_params = None
    gb.init(blocking=None)
    gb.init(blocking=None)
    gb._autoinit
