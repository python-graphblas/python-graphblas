"""Single-element access: the tightest hot loops in the library.

Extracting or assigning one element goes through the full expression + descriptor
machinery, so these times are almost entirely Python overhead. "hit" indexes an
element that is present; "miss" indexes an empty slot (the extract returns an
empty Scalar / ``None`` value).

Several of these paths gained dedicated fast lanes (single-element extract,
``get``, ``__contains__``, integer ``__setitem__``); the benchmarks below track
each one so a regression that re-routes through the slow expression path shows up.
Every ``setup`` warms the path once so the first timed sample is not inflated by
one-off numba/dtype/operator cache population.
"""

import graphblas as gb
from graphblas import Scalar

try:
    from . import common
except ImportError:  # imported flat by asv, not as a package
    import common


class ScalarObject:
    """Scalar construction and .value round-trips."""

    def setup(self):
        self.s = Scalar.from_value(3.14, dtype=gb.dtypes.FP64)
        # Warm construction / value round-trip caches.
        for _ in range(3):
            Scalar.from_value(3.14, dtype=gb.dtypes.FP64)
            self.s.value
            self.s.value = 2.0
            self.s.dup()

    def time_from_value(self):
        Scalar.from_value(3.14, dtype=gb.dtypes.FP64)

    def time_get_value(self):
        self.s.value

    def time_set_value(self):
        self.s.value = 2.0

    def time_dup(self):
        self.s.dup()


class VectorElement:
    """Get / extract / assign a single vector element, present vs missing."""

    def setup(self):
        # size 10000, ~1000 present entries; index 0 forced present, 1 forced empty
        self.v = common.make_vector(size=10_000, nnz=1_000, seed=10)
        self.v[0] = 1.0
        del self.v[1]
        self.hit = 0
        self.miss = 1
        # Warm each timed path once (extract, get, contains, setitem, the value /
        # float / int single-extract fast paths) so no timed sample pays the
        # first-call numba compile or cache-fill cost.
        for _ in range(3):
            self.v[self.hit].new()
            self.v[self.miss].new()
            self.v[self.hit].value
            self.v[self.miss].value
            float(self.v[self.hit])
            int(self.v[self.hit])
            self.v.get(self.hit)
            self.v.get(self.miss)
            self.v.get(self.miss, 0.0)
            _ = self.hit in self.v
            _ = self.miss in self.v
            self.v[self.hit] = 5.0

    def time_getitem_hit(self):
        self.v[self.hit].new()

    def time_getitem_miss(self):
        self.v[self.miss].new()

    def time_value_hit(self):
        self.v[self.hit].value

    def time_value_miss(self):
        self.v[self.miss].value

    def time_float_hit(self):
        float(self.v[self.hit])

    def time_int_hit(self):
        int(self.v[self.hit])

    def time_get_hit(self):
        self.v.get(self.hit)

    def time_get_miss(self):
        self.v.get(self.miss)

    def time_get_miss_default(self):
        # `get` with an explicit default; the miss returns the default rather than None.
        self.v.get(self.miss, 0.0)

    def time_contains_hit(self):
        self.hit in self.v

    def time_contains_miss(self):
        self.miss in self.v

    def time_setitem(self):
        # Overwrites an existing slot, so state is stable across repeated calls.
        self.v[self.hit] = 5.0


class MatrixElement:
    """Get / extract / assign a single matrix element, present vs missing."""

    def setup(self):
        self.M = common.make_matrix(n=10_000, nnz=1_000, seed=11)
        self.M[0, 0] = 1.0
        del self.M[1, 1]
        self.hit = (0, 0)
        self.miss = (1, 1)
        for _ in range(3):
            self.M[self.hit[0], self.hit[1]].new()
            self.M[self.miss[0], self.miss[1]].new()
            self.M[self.hit[0], self.hit[1]].value
            self.M[self.miss[0], self.miss[1]].value
            float(self.M[self.hit[0], self.hit[1]])
            self.M.get(self.hit[0], self.hit[1])
            self.M.get(self.miss[0], self.miss[1])
            self.M.get(self.miss[0], self.miss[1], 0.0)
            _ = self.hit in self.M
            _ = self.miss in self.M
            self.M[self.hit[0], self.hit[1]] = 5.0

    def time_getitem_hit(self):
        self.M[self.hit[0], self.hit[1]].new()

    def time_getitem_miss(self):
        self.M[self.miss[0], self.miss[1]].new()

    def time_value_hit(self):
        self.M[self.hit[0], self.hit[1]].value

    def time_value_miss(self):
        self.M[self.miss[0], self.miss[1]].value

    def time_float_hit(self):
        float(self.M[self.hit[0], self.hit[1]])

    def time_get_hit(self):
        self.M.get(self.hit[0], self.hit[1])

    def time_get_miss(self):
        self.M.get(self.miss[0], self.miss[1])

    def time_get_miss_default(self):
        self.M.get(self.miss[0], self.miss[1], 0.0)

    def time_contains_hit(self):
        self.hit in self.M

    def time_contains_miss(self):
        self.miss in self.M

    def time_setitem(self):
        self.M[self.hit[0], self.hit[1]] = 5.0
