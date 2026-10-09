"""Import and cold-start timing.

These use asv's ``timeraw_*`` form: each returns a code string that asv runs in a
*fresh* subprocess, so the measurement is the true cold cost (module already
imported into the benchmark process would otherwise read as ~0). ``timeraw``
benchmarks cannot see ``setup`` state or this module's imports, so everything
they need must be inside the returned string.

``graphblas`` initializes lazily: ``import graphblas`` is cheap, and the real
cost (loading the C library, building operator namespaces) is deferred until
first use. The staged benchmarks below separate those phases.
"""


class ImportTiming:
    # A little headroom: importing numba/llvmlite on first operator use is not fast.
    timeout = 120

    def timeraw_import_graphblas(self):
        return "import graphblas"

    def timeraw_from_import_core_types(self):
        return "from graphblas import Matrix, Vector, Scalar"

    def timeraw_import_then_init(self):
        # Force backend initialization (loads the C library).
        return "import graphblas as gb; gb.init('suitesparse')"

    def timeraw_first_operator_access(self):
        # First attribute access on an operator namespace triggers its lazy build.
        return "import graphblas as gb; gb.binary.plus"

    def timeraw_first_matrix(self):
        # End-to-end cold path: import, init, and build one tiny Matrix.
        return "import graphblas as gb; gb.Matrix.from_coo([0], [0], [1.0], nrows=1, ncols=1)"
