if __name__ == "__main__":
    import pytest

    import graphblas

    with pytest.raises(ValueError, match="Bad backend name"):
        graphblas.init("bad_name")

    graphblas.ffi
    graphblas.matrix
    graphblas.Matrix
    with pytest.raises(
        graphblas.exceptions.GraphblasException,
        match="graphblas objects accessed prior to manual initialization",
    ):
        graphblas.init()
    # hack it to test this edge case
    graphblas._init_params = None
    graphblas.init(blocking=None)
    graphblas.init(blocking=None)
