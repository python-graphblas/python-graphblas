from . import lib
from .base import recorder
from .mask import Mask
from .matrix import TransposedMatrix
from .ops import TypedOpBase


def gbstr(arg):
    """ Convert arg to a string as an argument in a GraphBLAS call"""
    if arg is None:
        return "NULL"
    elif isinstance(arg, TypedOpBase):
        name = arg.gb_name
    elif isinstance(arg, Mask):
        name = arg.mask.name
    elif type(arg) is TransposedMatrix:
        name = arg._matrix.name
    else:
        name = arg.name
    if not name:
        return f"temp_{type(arg).__name__.lower()}"
    return name


class Recorder:
    """Record GraphBLAS C calls.

    The recorder can use `.start()` and `.stop()` to enable/disable recording,
    or it can be used as a context manager.

    For example,

    >>> with Recorder() as rec:
    ...     C = A.mxm(B).new()
    >>> rec.data[0]
    'GrB_mxm(C, NULL, NULL, GxB_PLUS_TIMES_INT64, A, B, NULL)'

    Currently, only one recorder will record at a given time.
    """

    def __init__(self):
        self.data = []
        self._token = None

    def record(self, cfunc_name, args):
        if not hasattr(lib, cfunc_name):
            cfunc_name = f"GxB_{cfunc_name[4:]}"
        self.data.append(f'{cfunc_name}({", ".join(gbstr(x) for x in args)})')

    def start(self):
        if self._token is None:
            self._token = recorder.set(self)

    def stop(self):
        if self._token is not None:
            recorder.reset(self._token)
            self._token = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type_, value, traceback):
        self.stop()
