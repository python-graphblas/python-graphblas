import threading

from ...exceptions import check_status, check_status_carg
from .. import ffi, lib
from .config import BaseConfig

ffi_new = ffi.new


class ThreadLocal(threading.local):
    """Hold the active context for the current thread."""

    context = None


threadlocal = ThreadLocal()


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
        "gpu_id": -1,  # -1 means no GPU (I think)
    }

    def __init__(self, engage=True, *, nthreads=None, chunk=None, gpu_id=None):
        super().__init__()
        if nthreads is not None:
            self["nthreads"] = nthreads
        if chunk is not None:
            self["chunk"] = chunk
        if gpu_id is not None:
            self["gpu_id"] = gpu_id
        if engage:
            self.engage()

    def __new__(cls, engage=True, **opts):
        self = object.__new__(cls)
        self.gb_obj = ffi_new("GxB_Context*")
        check_status_carg(lib.GxB_Context_new(self.gb_obj), "Context", self.gb_obj[0])
        return self

    @classmethod
    def _from_obj(cls, gb_obj=None):
        self = object.__new__(cls)
        self.gb_obj = gb_obj
        self.__init__(engage=False)
        return self

    @classmethod
    def _maybe_new(cls):
        if threadlocal.context is not None:
            return threadlocal.context
        self = cls(engage=False)
        check_status(lib.GxB_Context_engage(self._carg), self)
        # Don't assign to threadlocal.context; instead, let it disengage upon going out of scope
        return self

    @property
    def _carg(self):
        return self.gb_obj[0]

    def __del__(self):
        gb_obj = getattr(self, "gb_obj", None)
        if gb_obj is not None and lib is not None:  # pragma: no branch (safety)
            try:
                lib.GxB_Context_disengage(gb_obj[0])
            finally:
                check_status(lib.GxB_Context_free(gb_obj), self)

    def engage(self):
        check_status(lib.GxB_Context_engage(self._carg), self)
        threadlocal.context = self

    def disengage(self):
        if threadlocal.context is self:
            threadlocal.context = None
        check_status(lib.GxB_Context_disengage(self._carg), self)

    def __enter__(self):
        self.engage()

    def __exit__(self, exc_type, exc, exc_tb):
        self.disengage()

    @property
    def _context(self):
        return self

    @_context.setter
    def _context(self, val):
        if val is not None:
            raise AttributeError("'_context' attribute is read-only")


class GlobalContext(Context):
    @property
    def _carg(self):
        return self.gb_obj

    def __del__(self):  # pragma: no cover (safety)
        pass


global_context = GlobalContext._from_obj(lib.GxB_CONTEXT_WORLD)
