"""Regression tests for gh-559: Matrix/Vector must not be born in a reference cycle.

Historically each Matrix and Vector stored ``self.ss = ss(self)``, and the ss
object (and its config) held ``_parent`` back to the object, so instances died
only via the cyclic garbage collector, not by reference counting. For large
matrices in a tight loop this deferred the release of the C-side GrB buffer
until gc happened to run. ``.ss`` is now built lazily per access and stored
nowhere, so the cycle never forms. These tests pin that behavior; they are
suitesparse-only because ``.ss`` exists only on that backend.
"""

import gc
import weakref

import numpy as np
import pytest

from graphblas import Matrix, Vector, backend, semiring
from graphblas.core.ss.matrix import ss as matrix_ss_class
from graphblas.core.ss.vector import ss as vector_ss_class

if backend != "suitesparse":
    pytest.skip("A.ss only available with suitesparse backend", allow_module_level=True)


def _dies_by_reference_count(factory):
    """True if the object dies the instant its last strong ref drops, with gc off."""
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        obj = factory()
        wref = weakref.ref(obj)
        del obj
        return wref() is None
    finally:
        if was_enabled:
            gc.enable()


def test_matrix_dies_by_reference_count():
    assert _dies_by_reference_count(lambda: Matrix(float, 5, 5))


def test_vector_dies_by_reference_count():
    assert _dies_by_reference_count(lambda: Vector(float, 5))


def test_mxm_result_dies_by_reference_count():
    A = Matrix.from_dense(np.arange(9.0).reshape(3, 3) + 1)
    B = Matrix.from_dense(np.arange(9.0).reshape(3, 3) + 1)
    assert _dies_by_reference_count(lambda: A.mxm(B, semiring.min_plus).new())


def test_ss_namespace_is_functional():
    A = Matrix.from_coo([0, 1, 2], [0, 1, 2], [1.0, 2.0, 3.0], nrows=3, ncols=3)
    # Built fresh each access (nothing is stored on the instance).
    assert A.ss is not A.ss
    # Introspection still works through a fresh access.
    assert A.ss.nbytes > 0
    matrix_formats = {
        "csr",
        "csc",
        "hypercsr",
        "hypercsc",
        "bitmapr",
        "bitmapc",
        "fullr",
        "fullc",
        "coor",
        "cooc",
    }
    assert A.ss.export()["format"] in matrix_formats
    # Config get then set then get, each through an independent `.ss`.
    assert A.ss.config["format"] in {"by_row", "by_col"}
    A.ss.config["format"] = "by_col"
    assert A.ss.config["format"] == "by_col"

    v = Vector.from_coo([0, 2], [1.0, 3.0], size=4)
    assert v.ss is not v.ss
    assert v.ss.nbytes > 0
    assert v.ss.export()["format"] in {"sparse", "bitmap", "full"}


def test_ss_class_access_returns_namespace_class():
    # Class-level access must still yield the ss class so its import_* classmethods work.
    assert Matrix.ss is matrix_ss_class
    assert Vector.ss is vector_ss_class
    assert hasattr(Matrix.ss, "import_any")
    assert hasattr(Vector.ss, "import_any")


def test_ss_attribute_is_read_only():
    A = Matrix(float, 3, 3)
    with pytest.raises(AttributeError):
        A.ss = 5
    v = Vector(float, 3)
    with pytest.raises(AttributeError):
        v.ss = 5


def test_views_have_working_ss():
    # A single-column Matrix cast to a Vector (_as_vector) is a view with _parent set.
    A = Matrix.from_coo([0, 1], [0, 0], [1.0, 2.0], nrows=3, ncols=1)
    v = A._as_vector()
    assert v._parent is A
    assert v.ss.nbytes > 0
    # A Vector cast to a Matrix (_as_matrix) is likewise a view.
    w = Vector.from_coo([0, 2], [1.0, 3.0], size=4)
    M = w._as_matrix()
    assert M._parent is w
    assert M.ss.nbytes > 0


def test_batched_mxm_loop_does_not_accumulate_matrices():
    # gh-559: with the cyclic collector switched off, a batched
    # mxm -> to_dense -> discard loop must not pile up Matrix objects.
    rng = np.random.default_rng(0)
    A = Matrix.from_dense(rng.random((16, 8)) + 0.1)
    B = Matrix.from_dense(rng.random((8, 24)) + 0.1)

    def live_matrices():
        return sum(1 for o in gc.get_objects() if type(o) is Matrix)

    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        # Warm up one iteration so any one-time caches are populated first.
        C = Matrix(float, 16, 24)
        C << A.mxm(B, semiring.min_plus)
        C.to_dense(0.0)
        del C
        baseline = live_matrices()
        for _ in range(50):
            C = Matrix(float, 16, 24)
            C << A.mxm(B, semiring.min_plus)
            C.to_dense(0.0)
            del C
        # Without the fix this would be baseline + 50.
        assert live_matrices() <= baseline
    finally:
        if was_enabled:
            gc.enable()
