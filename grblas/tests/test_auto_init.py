if __name__ == "__main__":
    import pytest
    import grblas

    grblas.ffi
    grblas.matrix
    grblas.Matrix
    with pytest.raises(
        grblas.exceptions.GrblasException,
        match="grblas objects accessed prior to manual initialization",
    ):
        grblas.init()
