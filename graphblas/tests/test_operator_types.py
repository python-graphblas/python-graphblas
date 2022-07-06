import itertools
from collections import defaultdict

from graphblas import binary, dtypes, monoid, operator, semiring, unary
from graphblas.dtypes import (
    BOOL,
    FP32,
    FP64,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
)

if dtypes._supports_complex:
    from graphblas.dtypes import FC32, FC64
else:
    FC32 = "FC32"
    FC64 = "FC64"

BOOL = frozenset({BOOL})
UINT = frozenset({UINT8, UINT16, UINT32, UINT64})
INT = frozenset({INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64})
BOOLINT = frozenset(BOOL | INT)
FP = frozenset({FP32, FP64})
FPINT = frozenset(FP | INT)
NOFC = frozenset(BOOL | FPINT)
FC = frozenset({FC32, FC64})
FCFP = frozenset(FC | FP)
ALL = frozenset(NOFC | FC)
POS = frozenset({INT32, INT64})
NOBOOL = frozenset(ALL - BOOL)
INT64 = frozenset({INT64})

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
    (NOFC, FP): {"cbrt", "erf", "erfc", "lgamma", "tgamma"},
    (NOFC, NOFC): {"lnot"},
}
BINARY = {
    (ALL, ALL): {
        "any", "cdiv", "first", "iseq", "isne", "minus", "pair", "plus", "pow", "rdiv",
        "rminus", "second", "times",
    },
    (ALL, BOOL): {"eq", "ne"},
    (ALL, FCFP): {"truediv", "rtruediv"},
    (ALL, NOBOOL): {"rpow"},  # different for pow, b/c rpow is a UDF
    (ALL, NOFC): {"absfirst", "abssecond"},
    (ALL, POS): {
        "firsti", "firsti1", "firstj", "firstj1", "secondi", "secondi1", "secondj", "secondj1",
    },
    (BOOLINT, FP): {"ldexp"},
    (INT, INT): {"band", "bclr", "bget", "bor", "bset", "bshift", "bxnor", "bxor"},
    (INT, INT64): {"binom"},
    (NOFC, BOOL): {"ge", "gt", "le", "lt", "lxnor"},
    (NOFC, FC): {"cmplx"},
    (NOFC, FP): {"atan2", "copysign", "fmod", "hypot", "remainder"},
    (NOFC, FPINT): {"floordiv", "rfloordiv"},
    (NOFC, NOFC): {"isge", "isgt", "isle", "islt", "land", "lor", "lxor", "max", "min"},
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
    (ALL, FCFP): [
        {"any", "plus", "times"},
        {"truediv", "rtruediv"},
    ],
    (ALL, NOBOOL): [
        {"any", "plus", "times"},
        {"rpow"},
    ],
    (ALL, NOFC): [
        {"max"},
        {"absfirst", "abssecond"},
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
    (NOFC, FP): [
        {"max", "min"},
        {"truediv", "rtruediv"},
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
        {"absfirst", "abssecond"},
    ],
    (NOFC, FPINT): [
        {"any", "max", "min", "plus", "times"},
        {"floordiv", "rfloordiv"},
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
    # Created by
    "plus_copysign",
    "lazy",
    "lazy2",
    "lazy_lazy",
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
    "udt_identity",
    "udt_any",
    "udt_semiring",
    "any_any",
    "unary_pickle",
    "binary_pickle",
    "monoid_pickle",
    "semiring_pickle",
    # numpy-graphblas commutation (can we clean this up?)
    "band_land",
    "band_lor",
    "band_lxor",
    "band_max",
    "band_min",
    "band_minus",
    "band_plus",
    "band_pow",
    "band_rpow",
    "band_times",
    "bor_land",
    "bor_lor",
    "bor_lxor",
    "bor_max",
    "bor_min",
    "bor_minus",
    "bor_plus",
    "bor_pow",
    "bor_rpow",
    "bor_times",
    "bxor_land",
    "bxor_lor",
    "bxor_lxor",
    "bxor_max",
    "bxor_min",
    "bxor_minus",
    "bxor_plus",
    "bxor_pow",
    "bxor_rpow",
    "bxor_times",
    "eq_max",
    "eq_min",
    "eq_plus",
    "eq_times",
    "land_max",
    "land_min",
    "land_minus",
    "land_plus",
    "land_pow",
    "land_rpow",
    "land_times",
    "lor_max",
    "lor_min",
    "lor_minus",
    "lor_plus",
    "lor_pow",
    "lor_rpow",
    "lor_times",
    "lxor_max",
    "lxor_min",
    "lxor_minus",
    "lxor_plus",
    "lxor_pow",
    "lxor_rpow",
    "lxor_times",
    "max_atan2",
    "max_band",
    "max_bor",
    "max_bxor",
    "max_copysign",
    "max_eq",
    "max_ge",
    "max_gt",
    "max_ldexp",
    "max_le",
    "max_lt",
    "max_ne",
    "max_pow",
    "max_rpow",
    "min_atan2",
    "min_band",
    "min_bor",
    "min_bxor",
    "min_copysign",
    "min_eq",
    "min_ge",
    "min_gt",
    "min_ldexp",
    "min_le",
    "min_lt",
    "min_ne",
    "min_pow",
    "min_rpow",
    "plus_atan2",
    "plus_band",
    "plus_bor",
    "plus_bxor",
    "plus_ldexp",
    "times_atan2",
    "times_band",
    "times_bor",
    "times_bxor",
    "times_copysign",
    "times_eq",
    "times_ge",
    "times_gt",
    "times_ldexp",
    "times_le",
    "times_lt",
    "times_ne",
    "times_pow",
    "times_rpow",
    "band_rminus",
    "bor_rminus",
    "bxor_rminus",
    "land_rminus",
    "lor_rminus",
    "lxor_rminus",
}


def _run_test(module, typ, expected):
    if not dtypes._supports_complex:
        # Merge keys with FC types removed
        d = defaultdict(set)
        for (k1, k2), val in expected.items():
            key = (k1 - FC, k2 - FC)
            if not key[0] or not key[1]:
                continue
            d[key] |= val
        expected = d
    seen = defaultdict(set)
    for name in dir(module):
        val = getattr(module, name)
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
            raise AssertionError("graphblas has more than expected:", sorted(extra))
        missing = expected_names - seen_names
        if missing:
            raise AssertionError("graphblas has less than expected:", sorted(missing))

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
