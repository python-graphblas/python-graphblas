import threading

from ...exceptions import InvalidValue, check_status, check_status_carg
from .. import ffi, lib
from . import _IS_SSGB7
from .config import BaseConfig

ffi_new = ffi.new
if _IS_SSGB7:
    # Context was introduced in SuiteSparse:GraphBLAS 8.0
    import suitesparse_graphblas as ssgb

    raise ImportError(
        "Context was added to SuiteSparse:GraphBLAS in version 8; "
        f"current version is {ssgb.__version__}"
    )


class Context(BaseConfig):
    _context_keys = {"chunk", "gpu_id", "nthreads"}
    _options = {
        "chunk": (lib.GxB_CONTEXT_CHUNK, "double"),
        "gpu_id": (lib.GxB_CONTEXT_GPU_ID, "int"),
        "nthreads": (lib.GxB_CONTEXT_NTHREADS, "int"),
    }
    _defaults = {
        "nthreads": 0,
        "chunk": 0,
        "gpu_id": -1,  # -1 means no GPU
    }

    def __init__(self, engage=True, *, stack=True, nthreads=None, chunk=None, gpu_id=None):
        super().__init__()
        self.gb_obj = ffi_new("GxB_Context*")
        check_status_carg(lib.GxB_Context_new(self.gb_obj), "Context", self.gb_obj[0])
        if stack:
            context = threadlocal.context
            self["nthreads"] = context["nthreads"] if nthreads is None else nthreads
            self["chunk"] = context["chunk"] if chunk is None else chunk
            self["gpu_id"] = context["gpu_id"] if gpu_id is None else gpu_id
        else:
            if nthreads is not None:
                self["nthreads"] = nthreads
            if chunk is not None:
                self["chunk"] = chunk
            if gpu_id is not None:
                self["gpu_id"] = gpu_id
        self._prev_context = None
        if engage:
            self.engage()

    @classmethod
    def _from_obj(cls, gb_obj=None):
        self = object.__new__(cls)
        self.gb_obj = gb_obj
        self._prev_context = None
        super().__init__(self)
        return self

    @property
    def _carg(self):
        return self.gb_obj[0]

    def dup(self, engage=True, *, nthreads=None, chunk=None, gpu_id=None):
        if nthreads is None:
            nthreads = self["nthreads"]
        if chunk is None:
            chunk = self["chunk"]
        if gpu_id is None:
            gpu_id = self["gpu_id"]
        return type(self)(engage, stack=False, nthreads=nthreads, chunk=chunk, gpu_id=gpu_id)

    def __del__(self):
        gb_obj = getattr(self, "gb_obj", None)
        if gb_obj is not None and lib is not None:  # pragma: no branch (safety)
            try:
                self.disengage()
            except InvalidValue:
                pass
            lib.GxB_Context_free(gb_obj)

    def engage(self):
        if self._prev_context is None and (context := threadlocal.context) is not self:
            self._prev_context = context
        check_status(lib.GxB_Context_engage(self._carg), self)
        threadlocal.context = self

    def _engage(self):
        """Like engage, but don't set to threadlocal.context.

        This is useful if you want to disengage when the object is deleted by going out of scope.
        """
        if self._prev_context is None and (context := threadlocal.context) is not self:
            self._prev_context = context
        check_status(lib.GxB_Context_engage(self._carg), self)

    def disengage(self):
        prev_context = self._prev_context
        self._prev_context = None
        if threadlocal.context is self:
            if prev_context is not None:
                threadlocal.context = prev_context
                prev_context.engage()
            else:
                threadlocal.context = global_context
                check_status(lib.GxB_Context_disengage(self._carg), self)
        elif prev_context is not None and threadlocal.context is prev_context:
            prev_context.engage()
        else:
            check_status(lib.GxB_Context_disengage(self._carg), self)

    def __enter__(self):
        self.engage()
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self.disengage()

    @property
    def _context(self):
        return self

    @_context.setter
    def _context(self, val):
        if val is not None and val is not self:
            raise AttributeError("'_context' attribute is read-only")


class GlobalContext(Context):
    @property
    def _carg(self):
        return self.gb_obj

    def __del__(self):  # pragma: no cover (safety)
        pass


global_context = GlobalContext._from_obj(lib.GxB_CONTEXT_WORLD)


class ThreadLocal(threading.local):
    """Hold the active context for the current thread."""

    context = global_context


threadlocal = ThreadLocal()
