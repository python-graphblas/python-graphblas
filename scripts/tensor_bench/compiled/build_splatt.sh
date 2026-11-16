#!/usr/bin/env bash
# Build SPLATT (https://github.com/ShadenSmith/splatt) and the kit's SPLATT driver.
#
#   conda activate graphblas-tensor-build     # environment-build.yml, or any env with its tools
#   bash scripts/tensor_bench/compiled/build_splatt.sh
#
# Clones SPLATT into $TENSOR_BENCH_BUILD (default: ../build), checks out the commit the
# maintainers measured, configures it with 64-bit indices and double precision, builds it with
# OpenMP, then compiles splatt_drv.c against its library into $TENSOR_BENCH_BUILD/bin.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=${TENSOR_BENCH_BUILD:-$HERE/../build}
PREFIX=${CONDA_PREFIX:?activate a conda environment with the build tools (environment-build.yml)}
SPLATT_COMMIT=${SPLATT_COMMIT:-6cb8628}

# Another activated environment may have exported its own compiler flags; use only this one's
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CMAKE_ARGS CPATH LIBRARY_PATH
export CC=${CC_FOR_SPLATT:-$PREFIX/bin/clang}
[ -x "$CC" ] || CC=$PREFIX/bin/cc
[ -x "$CC" ] || CC=cc
export CMAKE_PREFIX_PATH=$PREFIX
export CFLAGS="-isystem $PREFIX/include"
export LDFLAGS="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib"
case "$(uname -s)" in Darwin) SO=dylib ;; *) SO=so ;; esac

mkdir -p "$BUILD/bin"
cd "$BUILD"
[ -d splatt ] || git clone https://github.com/ShadenSmith/splatt.git
cd splatt
git checkout --quiet "$SPLATT_COMMIT"
./configure --cc="$CC" --index-size=64 --precision=double \
  --with-blas-lib="$PREFIX/lib/libblas.$SO" --with-lapack-lib="$PREFIX/lib/liblapack.$SO" \
  >"$BUILD/splatt_configure.log" 2>&1
OUTDIR=build/$(uname -s)-$(uname -m)
grep -q 'OpenMP_C_FLAGS:STRING=.' "$OUTDIR/CMakeCache.txt" ||
  echo "warning: OpenMP not found; SPLATT will run single-threaded" >&2
make -j "${JOBS:-8}" >"$BUILD/splatt_build.log" 2>&1

OMPFLAG=-fopenmp
"$CC" --version 2>/dev/null | grep -q clang && OMPFLAG=-fopenmp=libomp
"$CC" -isystem "$PREFIX/include" -O3 -std=c99 -fgnu89-inline -march=native "$OMPFLAG" -D_GNU_SOURCE -DNDEBUG \
  -Isrc -Iinclude "$HERE/splatt_drv.c" -o "$BUILD/bin/splatt_drv" "$OUTDIR/lib/libsplatt.a" \
  -L"$PREFIX/lib" -Wl,-rpath,"$PREFIX/lib" -lm "$PREFIX/lib/liblapack.$SO" "$PREFIX/lib/libblas.$SO"
echo "built $BUILD/bin/splatt_drv (and $BUILD/splatt/$OUTDIR/bin/splatt)"
