"""Guard the auto-generated expression surface against silent drift.

``graphblas/core/automethods.py`` is a generated-code module. Its name sets
(near line 347) drive ``scripts/autogenerate.py``, which copies auto-compute
properties onto the expression classes (``VectorExpression`` /
``VectorIndexExpr`` and the Scalar/Matrix equivalents) so that, for example,
``(A @ B).to_coo()`` works without a manual ``.new()`` first.

The trap this module closes: add a public method to ``Matrix``/``Vector``/
``Scalar``, forget to add its name to the sets and rerun the generator, and
nothing fails. The expression classes silently lack the method, but the concrete
types have it, so CI stays green.

The tests here assert, for each concrete type and for ``TransposedMatrix``:

1. Forward: every public method/property is EITHER reachable on the expression
   classes via auto-compute OR listed in ``OPT_OUT`` with a reason. Mutating
   methods, constructors, and cheap metadata deliberately do not auto-compute.
2. Reverse: every auto-generated name still exists on the concrete type (a
   rename that leaves a stale set entry is caught at import already, but this
   makes the failure legible).
3. Hygiene: every ``OPT_OUT`` entry is a live attribute that is not in fact
   covered, so the table cannot rot into stale excuses.

Coverage is derived at runtime from what the generator actually emitted onto the
expression classes (properties whose getter lives in the ``automethods`` module),
not from a second copy of the name sets. That keeps this test honest if the sets
are reorganized: only a real change in the generated surface moves coverage.

Dunder scope: value-forwarding dunders such as ``__getitem__``, ``__contains__``,
and ``__matmul__`` are generated and thus checked. Arithmetic and comparison
operator sugar such as ``__add__`` and ``__lt__`` is handled by the infix
expression system (``infixmethods.py``), not the value-forwarding automethods
path, so it is excluded from the dunder sweep by ``_OPERATOR_SUGAR_DUNDERS``.
Python object machinery is excluded via a baseline class so new
interpreter-version dunders never make this test flaky.
"""

from graphblas.core.matrix import (
    Matrix,
    MatrixExpression,
    MatrixIndexExpr,
    TransposedMatrix,
)
from graphblas.core.scalar import Scalar, ScalarExpression, ScalarIndexExpr
from graphblas.core.vector import Vector, VectorExpression, VectorIndexExpr

import pytest  # isort: skip

_AUTOMETHODS_MODULE = "graphblas.core.automethods"

# --- reasons an attribute deliberately does not auto-compute ----------------

_MUTATES = (
    "in-place mutator; auto-computing would mutate a throwaway temporary, "
    "not the caller's object"
)
_CONSTRUCTS = (
    "constructor (classmethod) that builds a new object from external data; "
    "not an accessor on a computed result"
)
_MATERIALIZES = (
    "produces a new concrete object; an expression is materialized with .new(), "
    "so an auto-compute property would be redundant"
)
_METADATA = (
    "cheap metadata known without materializing; exposed natively on expressions "
    "via BaseExpression, so it must not force a compute"
)
_SCALAR_STORAGE = (
    "storage-backing flag (C scalar vs GrB_Scalar); ScalarExpression and "
    "ScalarIndexExpr define it natively since they know how their result will "
    "materialize, so it must not force a compute"
)
_RAISES_MATERIALIZE = (
    "raises to require explicit materialization of the lazy transposed view "
    "(np.asarray / bool); mirrors Matrix-expression behavior"
)

# Every public name of a concrete type that intentionally is NOT auto-computed.
# Adding a public method almost never belongs here; it belongs in the automethods
# name sets. This table is for the deliberate exceptions only.
OPT_OUT = {
    "Scalar": {
        "clear": _MUTATES,
        "update": _MUTATES,
        "dup": _MATERIALIZES,
        "from_value": _CONSTRUCTS,
        "dtype": _METADATA,
        "ndim": _METADATA,
        "shape": _METADATA,
        "is_cscalar": _SCALAR_STORAGE,
        "is_grbscalar": _SCALAR_STORAGE,
    },
    "Vector": {
        "build": _MUTATES,
        "clear": _MUTATES,
        "resize": _MUTATES,
        "update": _MUTATES,
        "dup": _MATERIALIZES,
        "from_coo": _CONSTRUCTS,
        "from_dense": _CONSTRUCTS,
        "from_dict": _CONSTRUCTS,
        "from_pairs": _CONSTRUCTS,
        "from_scalar": _CONSTRUCTS,
        "dtype": _METADATA,
        "ndim": _METADATA,
        "shape": _METADATA,
        "size": _METADATA,
    },
    "Matrix": {
        "build": _MUTATES,
        "clear": _MUTATES,
        "resize": _MUTATES,
        "setdiag": _MUTATES,
        "update": _MUTATES,
        "dup": _MATERIALIZES,
        "from_coo": _CONSTRUCTS,
        "from_csc": _CONSTRUCTS,
        "from_csr": _CONSTRUCTS,
        "from_dcsc": _CONSTRUCTS,
        "from_dcsr": _CONSTRUCTS,
        "from_dense": _CONSTRUCTS,
        "from_dicts": _CONSTRUCTS,
        "from_edgelist": _CONSTRUCTS,
        "from_scalar": _CONSTRUCTS,
        "dtype": _METADATA,
        "ncols": _METADATA,
        "ndim": _METADATA,
        "nrows": _METADATA,
        "shape": _METADATA,
    },
    "TransposedMatrix": {
        "dup": _MATERIALIZES,
        "new": _MATERIALIZES,
        "dtype": _METADATA,
        "ncols": _METADATA,
        "ndim": _METADATA,
        "nrows": _METADATA,
        "shape": _METADATA,
        "__array__": _RAISES_MATERIALIZE,
        "__bool__": _RAISES_MATERIALIZE,
    },
}

# Concrete type + the expression classes the generator targets for it.
# TransposedMatrix has no expression class of its own; as a read-only Matrix view
# it shares Matrix's generated surface.
_REGISTRY = {
    "Scalar": (Scalar, (ScalarExpression, ScalarIndexExpr)),
    "Vector": (Vector, (VectorExpression, VectorIndexExpr)),
    "Matrix": (Matrix, (MatrixExpression, MatrixIndexExpr)),
    "TransposedMatrix": (TransposedMatrix, (MatrixExpression, MatrixIndexExpr)),
}


# --- dunder scope helpers ---------------------------------------------------


class _Baseline:
    __slots__ = ()


# Object/interpreter machinery dunders. Computed from a trivial class so that
# version-specific additions (e.g. __firstlineno__, __static_attributes__) are
# absorbed automatically rather than hard-coded.
_MACHINERY_DUNDERS = (
    set(dir(_Baseline))
    | set(vars(_Baseline))
    | {
        "__del__",
        "__weakref__",
        "__dict__",
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__setstate__",
        "__networkx_backend__",
        "__networkx_plugin__",
    }
)

# Arithmetic / comparison / item-mutation sugar. These build lazy infix
# expressions (or mutate in place) and are handled by infixmethods.py, not the
# value-forwarding automethods path, so they are out of scope for this sweep.
_OPERATOR_SUGAR_DUNDERS = {
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__truediv__",
    "__rtruediv__",
    "__floordiv__",
    "__rfloordiv__",
    "__mod__",
    "__rmod__",
    "__pow__",
    "__rpow__",
    "__divmod__",
    "__rdivmod__",
    "__xor__",
    "__rxor__",
    "__neg__",
    "__abs__",
    "__invert__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__setitem__",
    "__delitem__",
}


# --- introspection ----------------------------------------------------------


def _accessible(cls, name):
    """True if ``cls.name`` resolves without raising.

    ``dir()`` lists names that are not usable (e.g. ``ss`` under the
    suitesparse-vanilla backend raises on access); those are not part of the
    public surface a caller can rely on.
    """
    try:
        getattr(cls, name)
    except Exception:
        # Any failure to access means the name is not a usable public attribute.
        return False
    return True


def _public_surface(cls):
    return {n for n in dir(cls) if not n.startswith("_") and _accessible(cls, n)}


def _own_dunders(cls):
    return {n for n in vars(cls) if n.startswith("__") and n.endswith("__")}


def _generated_coverage(*expr_classes):
    """Names the generator emitted onto ``expr_classes``.

    A name counts as covered when the expression class exposes it as a property
    whose getter is defined in the automethods module, or as a callable copied
    from that module (the ``__iadd__``-style guards). This reads the actual
    generated surface, so it tracks the name sets without duplicating them.
    """
    names = set()
    for expr_class in expr_classes:
        for klass in expr_class.__mro__:
            for name, value in vars(klass).items():
                func = value.fget if isinstance(value, property) else value
                if callable(func) and getattr(func, "__module__", None) == _AUTOMETHODS_MODULE:
                    names.add(name)
    return names


def _coverage(label):
    _concrete, expr_classes = _REGISTRY[label]
    return _generated_coverage(*expr_classes)


def _fix_hint(label, names):
    _concrete, expr_classes = _REGISTRY[label]
    expr_names = " / ".join(c.__name__ for c in dict.fromkeys(expr_classes))
    return (
        f"{label} has public attribute(s) not reachable on its expression "
        f"classes ({expr_names}) and not listed in OPT_OUT[{label!r}]:\n"
        f"    {sorted(names)}\n\n"
        "If these should auto-compute on expressions, add each name to the "
        "matching set in graphblas/core/automethods.py (the sets near line 347) "
        "and regenerate with:\n"
        "    python scripts/autogenerate.py\n\n"
        "If they must NOT auto-compute (an in-place mutator, a constructor, or "
        f"cheap metadata), add them to OPT_OUT[{label!r}] in this file with a "
        "one-line reason."
    )


# --- tests ------------------------------------------------------------------


@pytest.mark.parametrize("label", list(_REGISTRY))
def test_public_methods_covered_or_opted_out(label):
    concrete, _expr_classes = _REGISTRY[label]
    coverage = _coverage(label)
    opt_out = OPT_OUT[label]
    uncovered = _public_surface(concrete) - coverage - set(opt_out)
    assert not uncovered, _fix_hint(label, uncovered)


@pytest.mark.parametrize("label", list(_REGISTRY))
def test_relevant_dunders_covered_or_opted_out(label):
    concrete, _expr_classes = _REGISTRY[label]
    coverage = _coverage(label)
    opt_out = OPT_OUT[label]
    candidates = (
        _own_dunders(concrete)
        - _MACHINERY_DUNDERS
        - _OPERATOR_SUGAR_DUNDERS
        - coverage
        - set(opt_out)
    )
    assert not candidates, _fix_hint(label, candidates)


@pytest.mark.parametrize("label", ["Scalar", "Vector", "Matrix"])
def test_no_stale_generated_names(label):
    concrete, _expr_classes = _REGISTRY[label]
    # _get_value is the auto-compute helper itself, not a mirror of a concrete
    # attribute, so it is expected not to exist on the concrete type.
    stale = {n for n in _coverage(label) if n != "_get_value" and not hasattr(concrete, n)}
    assert not stale, (
        f"Generated name(s) on the {label} expression classes no longer exist "
        f"on {label}: {sorted(stale)}. A concrete method was renamed or removed "
        "without updating the sets in graphblas/core/automethods.py; update the "
        "sets and rerun `python scripts/autogenerate.py`."
    )


@pytest.mark.parametrize("label", list(_REGISTRY))
def test_opt_out_entries_are_live(label):
    concrete, _expr_classes = _REGISTRY[label]
    opt_out = OPT_OUT[label]
    surface = _public_surface(concrete) | _own_dunders(concrete)
    missing = {n for n in opt_out if n not in surface}
    assert not missing, (
        f"OPT_OUT[{label!r}] lists name(s) that are not attributes of {label}: "
        f"{sorted(missing)}. Remove the stale entries (the method was renamed or "
        "removed)."
    )
    coverage = _coverage(label)
    redundant = {n for n in opt_out if n in coverage}
    assert not redundant, (
        f"OPT_OUT[{label!r}] lists name(s) that ARE auto-computed and so need no "
        f"opt-out: {sorted(redundant)}. Remove them from OPT_OUT."
    )


@pytest.mark.parametrize("label", list(_REGISTRY))
def test_opt_out_reasons_present(label):
    empty = {n for n, reason in OPT_OUT[label].items() if not reason or not reason.strip()}
    assert not empty, f"OPT_OUT[{label!r}] entries need a non-empty reason: {sorted(empty)}"
