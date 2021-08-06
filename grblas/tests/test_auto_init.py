if __name__ == "__main__":
    import pytest

    import grblas

    with pytest.raises(ValueError, match="Bad backend name"):
        grblas.init("bad_name")

    grblas.ffi
    grblas.matrix
    grblas.Matrix
    with pytest.raises(
        grblas.exceptions.GrblasException,
        match="grblas objects accessed prior to manual initialization",
    ):
        grblas.init()
