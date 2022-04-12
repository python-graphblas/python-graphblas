if __name__ == "__main__":
    import pytest
    import suitesparse_graphblas as ssgb

    import graphblas

    ssgb.initialize(blocking=False)

    with pytest.raises(RuntimeError, match="GraphBLAS has already been initialized with"):
        graphblas.init(blocking=True)
    graphblas.init(blocking=False)
