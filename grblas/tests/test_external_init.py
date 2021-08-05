if __name__ == "__main__":
    import pytest
    import suitesparse_graphblas as ssgb

    import grblas

    ssgb.initialize(blocking=False)

    with pytest.raises(RuntimeError, match="GraphBLAS has already been initialized with"):
        grblas.init(blocking=True)
    grblas.init(blocking=False)
