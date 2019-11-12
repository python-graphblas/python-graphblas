import pytest
from _grblas import ffi
from grblas import descriptor

def test_caching():
    """
    Test that building a descriptor is actually caching rather than building
    a new object for each call.
    """
    tocr = descriptor.build(output_replace=True, mask_complement=True, 
                            transpose_first=True, transpose_second=False)
    tocr2 = descriptor.build(output_replace=True, mask_complement=True, 
                             transpose_first=True, transpose_second=False)
    assert tocr is tocr2

def test_null_desc():
    """
    The default descriptor is not actually defined, but should be NULL
    """
    default = descriptor.build()
    assert default == ffi.NULL
