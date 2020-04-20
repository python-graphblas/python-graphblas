import itertools
from .. import ops
from . import binary, monoid
from .binary import _binary_names
from .monoid import _monoid_identities

_semiring_names = {
    f'{monoid_name}_{binary_name}'
    for monoid_name, binary_name in itertools.product(_monoid_identities, _binary_names)
}

# Remove incompatible combinations
# <non-int>_<int>
_semiring_names -= {
    f'{monoid_name}_{binary_name}'
    for monoid_name, binary_name in itertools.product(
        {'equal', 'hypot', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_or', 'logical_xor'},
        {'gcd', 'lcm', 'left_shift', 'right_shift'}
    )
}
# <non-float>_<float>
_semiring_names -= {
    f'{monoid_name}_{binary_name}'
    for monoid_name, binary_name in itertools.product(
        {'bitwise_and', 'bitwise_or', 'bitwise_xor', 'equal', 'gcd', 'logical_and', 'logical_or', 'logical_xor'},
        {'arctan2', 'copysign', 'divide', 'hypot', 'ldexp', 'logaddexp2', 'logaddexp', 'nextafter', 'true_divide'}
    )
}
# <float>_<non-float>
_semiring_names -= {
    f'{monoid_name}_{binary_name}'
    for monoid_name, binary_name in itertools.product(
        {'hypot', 'logaddexp', 'logaddexp2'},
        {'bitwise_and', 'bitwise_or', 'bitwise_xor'}
    )
}
# <bool>_<non-bool>
_semiring_names -= {
    f'{monoid_name}_{binary_name}'
    for monoid_name, binary_name in itertools.product(
        {'equal', 'logical_and', 'logical_or', 'logical_xor'},
        {'floor_divide', 'fmod', 'mod', 'power', 'remainder', 'subtract'}
    )
}
# <non-bool>_<bool>
_semiring_names -= {
    f'{monoid_name}_{binary_name}'
    for monoid_name, binary_name in itertools.product(
        {'gcd', 'hypot', 'logaddexp', 'logaddexp2'},
        {'equal', 'greater', 'greater_equal', 'less', 'less_equal',
         'logical_and', 'logical_or', 'logical_xor', 'not_equal'}
    )
}


def __dir__():
    return list(_semiring_names)


def __getattr__(name):
    if name not in _semiring_names:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    words = name.split('_')
    for i in range(1, len(words)):
        monoid_name = '_'.join(words[:i])
        if not hasattr(monoid, monoid_name):
            continue
        binary_name = '_'.join(words[i:])
        if hasattr(binary, binary_name):
            break
    semiring = ops.Semiring.register_anonymous(
        getattr(monoid, monoid_name),
        getattr(binary, binary_name),
        name
    )
    globals()[name] = semiring
    return semiring
