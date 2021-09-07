import itertools
from collections import defaultdict

from grblas import binary, dtypes, monoid, operator, semiring, unary

BOOL = frozenset({"BOOL"})
UINT = frozenset({"UINT8", "UINT16", "UINT32", "UINT64"})
INT = frozenset({"INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16", "UINT32", "UINT64"})
FP = frozenset({"FP32", "FP64"})
FPINT = frozenset(FP | INT)
NOFC = frozenset(BOOL | FPINT)
if dtypes._supports_complex:
    FC = frozenset({"FC32", "FC64"})
else:  # pragma: no cover
    FC = frozenset()
FCFP = frozenset(FC | FP)
ALL = frozenset(NOFC | FC)
POS = frozenset({"INT32", "INT64"})
NOBOOL = frozenset(ALL - BOOL)

# fmt: off
UNARY = {
    (ALL, ALL): {"ainv", "identity", "minv", "one"},
    (ALL, FCFP): {
        "acos", "acosh", "asin", "asinh", "atan", "atanh", "ceil", "cos", "cosh", "exp",
        "exp2", "expm1", "floor", "log", "log10", "log1p", "log2", "round", "signum",
        "sin", "sinh", "sqrt", "tan", "tanh", "trunc",
    },
    (ALL, NOFC): {"abs"},
    (ALL, POS): {"positioni", "positioni1", "positionj", "positionj1"},
    (FCFP, BOOL): {"isfinite", "isinf", "isnan"},
    (FC, FC): {"conj"},
    (FC, FP): {"carg", "cimag", "creal"},
    (FP, FP): {"frexpe", "frexpx"},
    (INT, INT): {"bnot"},
    (NOFC, FP): {"erf", "erfc", "lgamma", "tgamma"},
    (NOFC, NOFC): {"lnot"},
}
BINARY = {
    (ALL, ALL): {
        "any", "cdiv", "first", "iseq", "isne", "minus", "pair", "plus", "pow", "rdiv",
        "rminus", "second", "times",
    },
    (ALL, BOOL): {"eq", "ne"},
    (ALL, FP): {"truediv"},
    (ALL, NOFC): {"absfirst"},
    (ALL, POS): {
        "firsti", "firsti1", "firstj", "firstj1", "secondi", "secondi1", "secondj", "secondj1",
    },
    (INT, INT): {"band", "bclr", "bget", "bor", "bset", "bshift", "bxnor", "bxor"},
    (NOFC, BOOL): {"ge", "gt", "land", "le", "lor", "lt", "lxnor", "lxor"},
    (NOFC, FC): {"cmplx"},
    (NOFC, FP): {"atan2", "copysign", "fmod", "hypot", "ldexp", "remainder"},
    (NOFC, FPINT): {"floordiv"},
    (NOFC, NOFC): {"isge", "isgt", "isle", "islt", "max", "min"},
}
MONOID = {
    (UINT, UINT): {"band", "bor", "bxnor", "bxor"},
    (BOOL, BOOL): {"eq"},
    (NOFC, BOOL): {"land", "lor", "lxnor", "lxor"},  # others coerced to bool
    (NOFC, NOFC): {"max", "min"},  # max[bool] -> lor, min[bool] -> land
    (NOBOOL, NOBOOL): {"plus"},
    (ALL, ALL): {"any", "times"},  # times[bool] -> land
}

_SEMIRING1 = {
    (ALL, ALL): [
        {"any"},
        {"first", "pair", "second"},
    ],
    (ALL, FP): [
        {"any", "max", "min", "plus", "times"},
        {"truediv"},
    ],
    (ALL, NOFC): [
        {"max"},
        {"absfirst"},
    ],
    (ALL, POS): [  # POS, extra->INT64
        {"any", "max", "min", "plus", "times"},
        {"firsti", "firsti1", "firstj", "firstj1", "secondi", "secondi1", "secondj", "secondj1"},
    ],
    (ALL, BOOL): [  # BOOL, extra->BOOL
        {"eq", "land", "lor", "lxnor", "lxor"},
        {"first", "pair", "second"},
    ],
    (NOFC, BOOL): [  # BOOL, extra->BOOL, can't coerce FC here
        # BOOL `*_ne` uses `*_lxor`
        {"any", "eq", "land", "lor", "lxnor", "lxor"},
        {"eq", "land", "lor", "lxor", "ne", "ge", "gt", "le", "lt"},
    ],
    (FPINT, FPINT): [
        # Some of these are superseded by (ALL, ALL)
        # We could cast BOOL to INT8
        {"any", "max", "min", "plus", "times"},
        {
            "cdiv", "first", "iseq", "isge", "isgt", "isle", "islt", "isne", "land", "lor",
            "lxor", "max", "min", "minus", "pair", "plus", "rdiv", "rminus", "second", "times",
        },
    ],
    (INT, UINT): [  # signed int -> next larger unsigned int
        {"band", "bor", "bxnor", "bxor"},
        {"band", "bor", "bxnor", "bxor"},
    ],
    (NOBOOL, NOBOOL): [
        # Some of these are superseded by (ALL, ALL)
        # We could cast BOOL to INT8
        {"any", "plus", "times"},
        {"cdiv", "first", "minus", "pair", "plus", "pow", "rdiv", "rminus", "second", "times"},
    ],
    (NOBOOL, FPINT): [
        {"plus"},
        {"absfirst"},
    ],
    (NOFC, FPINT): [
        {"any", "max", "min", "plus", "times"},
        {"floordiv"},
    ],
    (NOFC, NOFC): [
        {"any", "min", "max"},
        {"land", "lor", "lxor", "first", "second"},
    ],
}
# fmt: on
_SEMIRING2 = defaultdict(lambda: (set(), set()))  # {semiring: [input_types, output_types]}
for key, (leftvals, rightvals) in _SEMIRING1.items():
    for left, right in itertools.product(leftvals, rightvals):
        name = f"{left}_{right}"
        if not hasattr(semiring, name):
            continue
        _SEMIRING2[name][0].update(key[0])
        _SEMIRING2[name][1].update(key[1])

SEMIRING = defaultdict(set)
for name, (ins, outs) in _SEMIRING2.items():
    SEMIRING[(frozenset(ins), frozenset(outs))].add(name)
IGNORE = {
    # UDFs created during tests
    "is_positive",
    "plus_one",
    "bin_test_func",
    "plus_plus_one",
    "plus_plus_two",
    "extra_twos",
    "myplus",
    "plus_myplus",
    "plus_numpy_copysign",
}


def _run_test(module, typ, expected):
    seen = defaultdict(set)
    for name, val in vars(module).items():
        if not isinstance(val, typ) or name in IGNORE:
            continue
        key = (frozenset(val.types.keys()), frozenset(val.types.values()))
        seen[key].add(name)
    seen = dict(seen)
    if seen != expected:  # pragma: no cover
        seen_names = set()
        for names in seen.values():
            seen_names.update(names)
        expected_names = set()
        for names in expected.values():
            expected_names.update(names)
        extra = seen_names - expected_names
        if extra:
            raise AssertionError("grblas has more than expected:", sorted(extra))
        missing = expected_names - seen_names
        if missing:
            raise AssertionError("grblas has less than expected:", sorted(missing))

        for key in sorted(seen.keys() - expected.keys()):
            print(
                "Item not expected: (%s, %s): %s"
                % (sorted(key[0]), sorted(key[1]), sorted(seen[key]))
            )
        for key in sorted(expected.keys() - seen.keys()):
            print(
                "Item expected, but not seen: (%s, %s): %s"
                % (sorted(key[0]), sorted(key[1]), sorted(expected[key]))
            )
        for key in sorted(expected.keys() & seen.keys()):
            if expected[key] != seen[key]:
                print("Bad match:", sorted(key[0]), sorted(key[1]))
                extra = seen[key] - expected[key]
                if extra:
                    print("    Extra:", sorted(extra))
                missing = expected[key] - seen[key]
                if missing:
                    print("    Missing:", sorted(missing))
        assert seen == expected


def test_unarytypes():
    _run_test(unary, operator.UnaryOp, UNARY)


def test_binarytypes():
    _run_test(binary, operator.BinaryOp, BINARY)


def test_monoid_types():
    _run_test(monoid, operator.Monoid, MONOID)


def test_semiring_types():
    _run_test(semiring, operator.Semiring, SEMIRING)
