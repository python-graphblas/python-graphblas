from collections.abc import Mapping

from ..core import ffi, lib
from ..core.base import _expect_type
from ..core.descriptor import lookup as descriptor_lookup
from ..core.matrix import Matrix, TransposedMatrix
from ..core.scalar import _as_scalar
from ..core.ss.config import BaseConfig
from ..core.ss.context import global_context
from ..core.ss.matrix import _concat_mn
from ..core.vector import Vector
from ..dtypes import INT64
from ..exceptions import _error_code_lookup


class _graphblas_ss:
    """Used in ``_expect_type``."""


_graphblas_ss.__name__ = "graphblas.ss"
_graphblas_ss = _graphblas_ss()


def diag(x, k=0, dtype=None, *, name=None, **opts):
    """GxB_Matrix_diag, GxB_Vector_diag.

    Extract a diagonal Vector from a Matrix, or construct a diagonal Matrix
    from a Vector.  Unlike ``Matrix.diag`` and ``Vector.diag``, this function
    returns a new object.

    Parameters
    ----------
    x : Vector or Matrix
        The Vector to assign to the diagonal, or the Matrix from which to
        extract the diagonal.
    k : int, default 0
        Diagonal in question.  Use ``k>0`` for diagonals above the main diagonal,
        and ``k<0`` for diagonals below the main diagonal.

    See Also
    --------
    Vector.diag
    Matrix.diag
    Vector.ss.build_diag
    Matrix.ss.build_diag

    """
    x = _expect_type(
        _graphblas_ss, x, (Matrix, TransposedMatrix, Vector), within="diag", argname="x"
    )
    k = _as_scalar(k, INT64, is_cscalar=True)
    if dtype is None:
        dtype = x.dtype
    typ = type(x)
    if typ is Vector:
        if opts:
            # Ignore opts for now
            desc = descriptor_lookup(**opts)  # noqa: F841 (keep desc in scope for context)
        size = x._size + abs(k.value)
        rv = Matrix(dtype, nrows=size, ncols=size, name=name)
        rv.ss.build_diag(x, k)
    else:
        if k.value < 0:
            size = max(0, min(x._nrows + k.value, x._ncols))
        else:
            size = max(0, min(x._ncols - k.value, x._nrows))
        rv = Vector(dtype, size=size, name=name)
        rv.ss.build_diag(x, k, **opts)
    return rv


def concat(tiles, dtype=None, *, name=None, **opts):
    """GxB_Matrix_concat.

    Concatenate a 2D list of Matrix objects into a new Matrix, or a 1D list of
    Vector objects into a new Vector.  To concatenate into existing objects,
    use ``Matrix.ss.concat`` or ``Vector.ss.concat``.

    Vectors may be used as ``Nx1`` Matrix objects when creating a new Matrix.

    This performs the opposite operation as ``split``.

    See Also
    --------
    Matrix.ss.split
    Matrix.ss.concat
    Vector.ss.split
    Vector.ss.concat

    """
    tiles, m, n, is_matrix = _concat_mn(tiles)
    if is_matrix:
        if dtype is None:
            dtype = tiles[0][0].dtype
        nrows = sum(row_tiles[0]._nrows for row_tiles in tiles)
        ncols = sum(tile._ncols for tile in tiles[0])
        rv = Matrix(dtype, nrows=nrows, ncols=ncols, name=name)
        rv.ss._concat(tiles, m, n, opts)
    else:
        if dtype is None:
            dtype = tiles[0].dtype
        size = sum(tile._nrows for tile in tiles)
        rv = Vector(dtype, size=size, name=name)
        rv.ss._concat(tiles, m, opts)
    return rv


class GlobalConfig(BaseConfig):
    """Get and set global configuration options for SuiteSparse:GraphBLAS.

    See SuiteSparse:GraphBLAS documentation for more details.

    Config parameters
    -----------------
    format : str, {"by_row", "by_col"}
        Rowwise or columnwise orientation
    hyper_switch : double
        Threshold that determines when to switch to hypersparse format
    bitmap_switch : List[double]
        Threshold that determines when to switch to bitmap format
    nthreads : int
        Maximum number of OpenMP threads to use. None, 0 or negative restores the
        default: the setting right after GraphBLAS was initialized, which GraphBLAS
        takes from OpenMP (it follows ``OMP_NUM_THREADS``). If GraphBLAS was
        initialized before python-graphblas, it is the setting python-graphblas found.
    chunk : double
        Control the number of threads used for small problems.
        For example, ``nthreads = floor(work / chunk)``.
    burble : bool
        Enable diagnostic printing from SuiteSparse:GraphBLAS
    print_1based : bool
    gpu_id : int
        Which GPU to use; default is -1, which means do not run on the GPU.
        Only available for SuiteSparse:GraphBLAS 10.1 and earlier.
        **GPU support is a work in progress; not recommended to use**
    jit_c_control : {"off", "pause", "run", "load", "on"}
        Control the CPU JIT:
        "off" : do not use the JIT and free all JIT kernels if loaded
        "pause" : do not run JIT kernels, but keep any loaded
        "run" : run JIT kernels if already loaded, but don't load or compile
        "load" : able to load and run JIT kernels; may not compile
        "on" : full JIT: able to compile, load, and run
        python-graphblas sets it at init from ``graphblas.config["jit_c_control"]``,
        or to "run" when that is not set, and raises that default "run" to "on" when
        C source needs compiling. Setting None restores the default. The rules are
        in the Technical Notes, "The JIT control"; the user guide's initialization
        page covers the trade-offs, and when "on" is safe.
    jit_use_cmake : bool
        Whether to use cmake to compile the JIT kernels.
        Always True for SuiteSparse built with MSVC (on Windows).
    jit_c_compiler_name : str
        C compiler for JIT kernels.
    jit_c_compiler_flags : str
        Flags for the C compiler.
    jit_c_linker_flags : str
        Link flags for the C compiler
    jit_c_libraries : str
        Libraries to link against.
    jit_c_cmake_libs : str
        Libraries to link against when cmake is used.
    jit_c_preface : str
        C code as preface to JIT kernels.
    jit_error_log : str
        Error log file.
    jit_cache_path : str
        The folder with the compiled kernels.

    Setting values to None restores the default value for most configurations.
    """

    _get_function = "GxB_Global_Option_get"
    _set_function = "GxB_Global_Option_set"
    _context_keys = {"chunk", "gpu_id", "nthreads"}
    _null_valid = {"bitmap_switch"}
    _options = {
        # Matrix/Vector format
        "hyper_switch": (lib.GxB_HYPER_SWITCH, "double"),
        "bitmap_switch": (lib.GxB_BITMAP_SWITCH, f"double[{lib.GxB_NBITMAP_SWITCH}]"),
        "format": (lib.GxB_FORMAT, "GxB_Format_Value"),
        # OpenMP control
        "nthreads": (lib.GxB_GLOBAL_NTHREADS, "int"),
        "chunk": (lib.GxB_GLOBAL_CHUNK, "double"),
        # Memory pool control
        # "memory_pool": (lib.GxB_MEMORY_POOL, "int64_t[64]"),  # No longer used
        # Diagnostics (skipping "printf" and "flush" for now)
        "burble": (lib.GxB_BURBLE, "bool"),
        "print_1based": (lib.GxB_PRINT_1BASED, "bool"),
        # JIT control
        "jit_c_control": (lib.GxB_JIT_C_CONTROL, "int"),
        "jit_use_cmake": (lib.GxB_JIT_USE_CMAKE, "bool"),
        "jit_c_compiler_name": (lib.GxB_JIT_C_COMPILER_NAME, "char*"),
        "jit_c_compiler_flags": (lib.GxB_JIT_C_COMPILER_FLAGS, "char*"),
        "jit_c_linker_flags": (lib.GxB_JIT_C_LINKER_FLAGS, "char*"),
        "jit_c_libraries": (lib.GxB_JIT_C_LIBRARIES, "char*"),
        "jit_c_cmake_libs": (lib.GxB_JIT_C_CMAKE_LIBS, "char*"),
        "jit_c_preface": (lib.GxB_JIT_C_PREFACE, "char*"),
        "jit_error_log": (lib.GxB_JIT_ERROR_LOG, "char*"),
        "jit_cache_path": (lib.GxB_JIT_CACHE_PATH, "char*"),
    }
    # GxB_GLOBAL_GPU_ID existed in SS:GraphBLAS <=10.1 as a scalar int.
    # It was renamed to GxB_GLOBAL_GPU_IDS in 10.2 and changed to a list type.
    if hasattr(lib, "GxB_GLOBAL_GPU_ID"):
        _options["gpu_id"] = (lib.GxB_GLOBAL_GPU_ID, "int")
    # Values to restore defaults
    _defaults = {
        "hyper_switch": lib.GxB_HYPER_DEFAULT,
        "bitmap_switch": None,
        "format": lib.GxB_FORMAT_DEFAULT,
        "nthreads": 0,  # See core.ss.config._nthreads_default: SuiteSparse takes 0 as 1 thread
        "chunk": 0,
        "burble": 0,
        "print_1based": 0,
        "gpu_id": -1,  # -1 means no GPU
    }
    _enumerations = {
        "format": {
            "by_row": lib.GxB_BY_ROW,
            "by_col": lib.GxB_BY_COL,
            # "no_format": lib.GxB_NO_FORMAT,  # Used by iterators; not valid here
        },
        "jit_c_control": {
            "off": lib.GxB_JIT_OFF,
            "pause": lib.GxB_JIT_PAUSE,
            "run": lib.GxB_JIT_RUN,
            "load": lib.GxB_JIT_LOAD,
            "on": lib.GxB_JIT_ON,
        },
    }

    def __setitem__(self, key, val):
        key = key.lower()
        if key != "jit_c_control":
            super().__setitem__(key, val)
            return
        from ..core.ss import jit_config

        # A 'run' written here is the user's, which _enable_jit_for_udt keeps, when
        # graphblas.config set a control at init. With nothing configured, it is the
        # default 'run', which is raised for C source, so saving and restoring the
        # default changes nothing. None is always the default.
        super().__setitem__(key, "run" if val is None else val)
        if isinstance(val, str):
            val = val.lower()
        elif val is not None:
            val = self._reverse_enumerations[key].get(val, val)
        jit_config._user_set_run = val == "run" and jit_config._configured_jit_c_control is not None


class About(Mapping):
    _modes = {
        lib.GrB_NONBLOCKING: "nonblocking",
        lib.GrB_BLOCKING: "blocking",
    }
    # GxB_NONBLOCKING_GPU and GxB_BLOCKING_GPU were modes of their own in SS:GraphBLAS <=10.5.1.
    # In 10.5.2 they are macros for GrB_NONBLOCKING and GrB_BLOCKING, which
    # python-suitesparse-graphblas does not export. Should they be aliases, `setdefault`
    # keeps the names "nonblocking" and "blocking".
    if hasattr(lib, "GxB_NONBLOCKING_GPU"):
        _modes.setdefault(lib.GxB_NONBLOCKING_GPU, "nonblocking_gpu")
        _modes.setdefault(lib.GxB_BLOCKING_GPU, "blocking_gpu")
    _mode_options = {
        "mode": lib.GxB_MODE,
    }
    _int3_options = {
        "library_version": lib.GxB_LIBRARY_VERSION,
        "api_version": lib.GxB_API_VERSION,
        "compiler_version": lib.GxB_COMPILER_VERSION,
    }
    _str_options = {
        "library_name": lib.GxB_LIBRARY_NAME,
        "library_date": lib.GxB_LIBRARY_DATE,
        "library_about": lib.GxB_LIBRARY_ABOUT,
        "library_url": lib.GxB_LIBRARY_URL,
        "library_license": lib.GxB_LIBRARY_LICENSE,
        "library_compile_date": lib.GxB_LIBRARY_COMPILE_DATE,
        "library_compile_time": lib.GxB_LIBRARY_COMPILE_TIME,
        "api_date": lib.GxB_API_DATE,
        "api_about": lib.GxB_API_ABOUT,
        "api_url": lib.GxB_API_URL,
        "compiler_name": lib.GxB_COMPILER_NAME,
    }
    _bool_options = {
        "openmp": lib.GxB_LIBRARY_OPENMP,
    }

    def __getitem__(self, key):
        key = key.lower()
        if key in self._mode_options:
            val_ptr = ffi.new("int32_t*")
            info = lib.GxB_Global_Option_get_INT32(self._mode_options[key], val_ptr)
            if info == lib.GrB_SUCCESS:  # pragma: no branch (safety)
                val = val_ptr[0]
                if val not in self._modes:  # pragma: no cover (sanity)
                    raise ValueError(f"Unknown mode: {val}")
                return self._modes[val]
        elif key in self._int3_options:
            val_ptr = ffi.new("int32_t[3]")
            info = lib.GxB_Global_Option_get_INT32(self._int3_options[key], val_ptr)
            if info == lib.GrB_SUCCESS:  # pragma: no branch (safety)
                return (val_ptr[0], val_ptr[1], val_ptr[2])
        elif key in self._str_options:
            val_ptr = ffi.new("char**")
            info = lib.GxB_Global_Option_get_CHAR(self._str_options[key], val_ptr)
            if info == lib.GrB_SUCCESS:  # pragma: no branch (safety)
                return ffi.string(val_ptr[0]).decode()
        elif key in self._bool_options:
            val_ptr = ffi.new("int32_t*")
            info = lib.GxB_Global_Option_get_INT32(self._bool_options[key], val_ptr)
            if info == lib.GrB_SUCCESS:  # pragma: no branch (safety)
                return bool(val_ptr[0])
        else:
            raise KeyError(key)
        raise _error_code_lookup[info](f"Failed to get info for {key}")  # pragma: no cover (safety)

    def __setattr__(self, key, value):
        # About never sets instance attributes, so an attribute write such as
        # ``about.mode = ...`` is a mistake, which would otherwise pass silently.
        if key.lower() in self:
            raise AttributeError(f"About option {key.lower()!r} is read-only")
        raise AttributeError(f"Unknown About option {key!r}; known options are {sorted(self)}.")

    def __iter__(self):
        return iter(
            sorted(
                self._mode_options.keys()
                | self._int3_options.keys()
                | self._str_options.keys()
                | self._bool_options.keys()
            )
        )

    def __len__(self):
        return (
            len(self._mode_options)
            + len(self._int3_options)
            + len(self._str_options)
            + len(self._bool_options)
        )

    __repr__ = GlobalConfig.__repr__
    _ipython_key_completions_ = GlobalConfig._ipython_key_completions_


about = About()
config = GlobalConfig(context=global_context)
