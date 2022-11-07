if __name__ == "__main__":
    import pytest
    import suitesparse_graphblas as ssgb

    import graphblas as gb

    ssgb.initialize(blocking=False)

    with pytest.raises(RuntimeError, match="GraphBLAS has already been initialized with"):
        gb.init(blocking=True)
    gb.init(blocking=False)
