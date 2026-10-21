#!/usr/bin/env bash
# Build TACO (https://github.com/tensor-compiler/taco) and the kit's TACO driver.
#
#   conda activate graphblas-tensor-build     # environment-build.yml, or any env with its tools
#   bash scripts/tensor_bench/compiled/build_taco.sh
#
# Clones TACO into $TENSOR_BENCH_BUILD (default: ../build), checks out the commit the
# maintainers measured, builds the library with OpenMP (Release), then compiles taco_drv.cpp
# against it into $TENSOR_BENCH_BUILD/bin, with the settings TACO needs at run time (it compiles
# its generated kernels with $TACO_CC) in taco_env.json next to it.
#
# Why a driver: TACO's command-line tool cannot run kernels at this commit. Its benchmark path
# compiles a module whose C source defines `compute` twice ("redefinition of 'compute'").
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=${TENSOR_BENCH_BUILD:-$HERE/../build}
PREFIX=${CONDA_PREFIX:?activate a conda environment with the build tools (environment-build.yml)}
TACO_COMMIT=${TACO_COMMIT:-0e79acb5}

unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CMAKE_ARGS
CC=${CC_FOR_TACO:-$PREFIX/bin/clang}
CXX=${CXX_FOR_TACO:-$PREFIX/bin/clang++}
[ -x "$CC" ] || CC=$PREFIX/bin/cc
[ -x "$CXX" ] || CXX=$PREFIX/bin/c++
export CPATH=$PREFIX/include
export LIBRARY_PATH=$PREFIX/lib
export LDFLAGS="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib"

mkdir -p "$BUILD/bin"
cd "$BUILD"
[ -d taco ] || git clone https://github.com/tensor-compiler/taco.git
cd taco
git checkout --quiet "$TACO_COMMIT"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOPENMP=ON -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" >"$BUILD/taco_configure.log" 2>&1
cmake --build build -j "${JOBS:-8}" --target taco-tool >"$BUILD/taco_build.log" 2>&1

OMPLIB=-lgomp
"$CXX" --version 2>/dev/null | grep -q clang && OMPLIB=-lomp
"$CXX" -std=c++17 -O2 -isystem "$PREFIX/include" -Iinclude -Isrc "$HERE/taco_drv.cpp" -o "$BUILD/bin/taco_drv" \
  -Lbuild/lib -ltaco -Wl,-rpath,"$BUILD/taco/build/lib" -L"$PREFIX/lib" -Wl,-rpath,"$PREFIX/lib" "$OMPLIB"
# TACO compiles its generated kernels at run time with $TACO_CC; compiled_runner.py reads
# these settings (TACO's own default flags, plus where omp.h and libomp are)
cat >"$BUILD/bin/taco_env.json" <<JSON
{"TACO_CC": "$CC", "CPATH": "$PREFIX/include", "LIBRARY_PATH": "$PREFIX/lib",
 "TACO_CFLAGS": "-O3 -ffast-math -std=c99 -Wl,-rpath,$PREFIX/lib"}
JSON
echo "built $BUILD/bin/taco_drv (kernel compiler settings in $BUILD/bin/taco_env.json)"
