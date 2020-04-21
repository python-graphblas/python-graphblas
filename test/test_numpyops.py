import pytest
import itertools
import grblas
import grblas.binary.numpy as npbinary
import grblas.monoid.numpy as npmonoid
import grblas.semiring.numpy as npsemiring


@pytest.mark.slow
def test_npsemiring():
    # This is a very slow test, since it forces creation of all numpy binary, monoid, and semiring objects
    for monoid_name, binary_name in itertools.product(
        sorted(npmonoid._monoid_identities),
        sorted(npbinary._binary_names)
    ):
        monoid = getattr(npmonoid, monoid_name)
        binary = getattr(npbinary, binary_name)
        name = monoid.name.split(".")[-1] + "_" + binary.name.split(".")[-1]
        semiring = grblas.ops.Semiring.register_anonymous(monoid, binary, name)
        if len(semiring.types) == 0:
            assert not hasattr(npsemiring, semiring.name)
        else:
            assert hasattr(npsemiring, semiring.name)
