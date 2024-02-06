from collections.abc import Mapping

from ..core import ffi, lib
from ..core.base import _expect_type
from ..core.descriptor import lookup as descriptor_lookup
from ..core.matrix import Matrix, TransposedMatrix
from ..core.scalar import _as_scalar
from ..core.ss import _IS_SSGB7
from ..core.ss.config import BaseConfig
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
        Maximum number of OpenMP threads to use
    chunk : double
        Control the number of threads used for small problems.
        For example, ``nthreads = floor(work / chunk)``.
    burble : bool
        Enable diagnostic printing from SuiteSparse:GraphBLAS
    print_1based : bool
    gpu_control : str, {"always", "never"}
        Only available for SuiteSparse:GraphBLAS 7
        **GPU support is a work in progress--not recommended to use**
    gpu_chunk : double
        Only available for SuiteSparse:GraphBLAS 7
        **GPU support is a work in progress--not recommended to use**
    gpu_id : int
        Which GPU to use; default is -1, which means do not run on the GPU.
        Only available for SuiteSparse:GraphBLAS >=8
        **GPU support is a work in progress--not recommended to use**
    jit_c_control : {"off", "pause", "run", "load", "on}
        Control the CPU JIT:
        "off" : do not use the JIT and free all JIT kernels if loaded
        "pause" : do not run JIT kernels, but keep any loaded
        "run" : run JIT kernels if already loaded, but don't load or compile
        "load" : able to load and run JIT kernels; may not compile
        "on" : full JIT: able to compile, load, and run
        Only available for SuiteSparse:GraphBLAS >=8
    jit_use_cmake : bool
        Whether to use cmake to compile the JIT kernels.
        Only available for SuiteSparse:GraphBLAS >=8
    jit_c_compiler_name : str
        C compiler for JIT kernels.
        Only available for SuiteSparse:GraphBLAS >=8
    jit_c_compiler_flags : str
        Flags for the C compiler.
        Only available for SuiteSparse:GraphBLAS >=8
    jit_c_linker_flags : str
        Link flags for the C compiler
        Only available for SuiteSparse:GraphBLAS >=8
    jit_c_libraries : str
        Libraries to link against.
        Only available for SuiteSparse:GraphBLAS >=8
    jit_c_cmake_libs : str
        Libraries to link against when cmake is used.
        Only available for SuiteSparse:GraphBLAS >=8
    jit_c_preface : str
        C code as preface to JIT kernels.
        Only available for SuiteSparse:GraphBLAS >=8
    jit_error_log : str
        Error log file.
        Only available for SuiteSparse:GraphBLAS >=8
    jit_cache_path : str
        The folder with the compiled kernels.
        Only available for SuiteSparse:GraphBLAS >=8

    Setting values to None restores the default value for most configurations.
    """

    _get_function = "GxB_Global_Option_get"
    _set_function = "GxB_Global_Option_set"
    if not _IS_SSGB7:
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
    }
    if _IS_SSGB7:
        _options.update(
            {
                "gpu_control": (lib.GxB_GLOBAL_GPU_CONTROL, "GrB_Desc_Value"),
                "gpu_chunk": (lib.GxB_GLOBAL_GPU_CHUNK, "double"),
            }
        )
    else:
        _options.update(
            {
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
                # CUDA GPU control
                "gpu_id": (lib.GxB_GLOBAL_GPU_ID, "int"),
            }
        )
    # Values to restore defaults
    _defaults = {
        "hyper_switch": lib.GxB_HYPER_DEFAULT,
        "bitmap_switch": None,
        "format": lib.GxB_FORMAT_DEFAULT,
        "nthreads": 0,
        "chunk": 0,
        "burble": 0,
        "print_1based": 0,
    }
    if not _IS_SSGB7:
        _defaults["gpu_id"] = -1  # -1 means no GPU
    _enumerations = {
        "format": {
            "by_row": lib.GxB_BY_ROW,
            "by_col": lib.GxB_BY_COL,
            # "no_format": lib.GxB_NO_FORMAT,  # Used by iterators; not valid here
        },
    }
    if _IS_SSGB7:
        _enumerations["gpu_control"] = {
            "always": lib.GxB_GPU_ALWAYS,
            "never": lib.GxB_GPU_NEVER,
        }
    else:
        _enumerations["jit_c_control"] = {
            "off": lib.GxB_JIT_OFF,
            "pause": lib.GxB_JIT_PAUSE,
            "run": lib.GxB_JIT_RUN,
            "load": lib.GxB_JIT_LOAD,
            "on": lib.GxB_JIT_ON,
        }


class About(Mapping):
    _modes = {
        lib.GrB_NONBLOCKING: "nonblocking",
        lib.GrB_BLOCKING: "blocking",
        lib.GxB_NONBLOCKING_GPU: "nonblocking_gpu",
        lib.GxB_BLOCKING_GPU: "blocking_gpu",
    }
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
if _IS_SSGB7:
    config = GlobalConfig()
else:
    # Context was introduced in SuiteSparse:GraphBLAS 8.0
    from ..core.ss.context import global_context

    config = GlobalConfig(context=global_context)
