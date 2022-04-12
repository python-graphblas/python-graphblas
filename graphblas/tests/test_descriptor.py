from graphblas import descriptor, lib


def test_caching():
    """
    Test that building a descriptor is actually caching rather than building
    a new object for each call.
    """
    tocr = descriptor.lookup(
        output_replace=True,
        mask_complement=True,
        mask_structure=True,
        transpose_first=True,
        transpose_second=False,
    )
    assert tocr.gb_obj == lib.GrB_DESC_RSCT0


def test_null_desc():
    """
    The default descriptor is not actually defined, but should be NULL
    """
    default = descriptor.lookup()
    assert default is None
