"""Index parsing overhead: the ``parse_index`` int fast lane.

Every ``v[i]`` / ``A[i, j]`` builds an ``IndexerResolver`` that runs each index
through ``parse_index``. A plain Python ``int`` takes a dedicated fast lane that
skips two ``np.issubdtype`` checks (~110ns each) used by the numpy-integer lane.
These benchmarks isolate that parsing cost from element extraction (no ``.new()``,
no value read), so a regression in the fast lane, or an index accidentally
falling out of it, is visible on its own.

Two granularities are measured:

* the public path (``v[i]`` returns an index expression without resolving it), and
* ``IndexerResolver(obj, idx)`` directly, the tightest view of ``parse_index``.

The plain-int and numpy-int variants sit side by side so the fast lane's margin
over the general lane is tracked directly.
"""

import numpy as np

from graphblas.core.expr import IndexerResolver

try:
    from . import common
except ImportError:  # imported flat by asv, not as a package
    import common


class VectorIndexParse:
    """Parse a single vector index: plain-int fast lane vs numpy-int lane."""

    def setup(self):
        self.v = common.make_vector(size=10_000, nnz=1_000, seed=12)
        self.i = 4321  # in range, positive
        self.neg = -1  # in range, triggers the negative-wrap branch
        self.npi = np.int64(4321)
        for _ in range(3):
            self.v[self.i]
            self.v[self.neg]
            self.v[self.npi]
            IndexerResolver(self.v, self.i)
            IndexerResolver(self.v, self.npi)

    # Public path: build the index expression (parse_index + expression object).
    def time_getitem_int(self):
        self.v[self.i]

    def time_getitem_int_negative(self):
        self.v[self.neg]

    def time_getitem_numpy_int(self):
        self.v[self.npi]

    # Tightest view: just the resolver (parse_index, no expression object).
    def time_resolver_int(self):
        IndexerResolver(self.v, self.i)

    def time_resolver_numpy_int(self):
        IndexerResolver(self.v, self.npi)


class MatrixIndexParse:
    """Parse a two-axis matrix index: plain-int fast lane vs numpy-int lane."""

    def setup(self):
        self.M = common.make_matrix(n=10_000, nnz=1_000, seed=13)
        self.ij = (4321, 8765)
        self.neg = (-1, -1)
        self.npij = (np.int64(4321), np.int64(8765))
        for _ in range(3):
            self.M[self.ij[0], self.ij[1]]
            self.M[self.neg[0], self.neg[1]]
            self.M[self.npij[0], self.npij[1]]
            IndexerResolver(self.M, self.ij)
            IndexerResolver(self.M, self.npij)

    def time_getitem_int(self):
        self.M[self.ij[0], self.ij[1]]

    def time_getitem_int_negative(self):
        self.M[self.neg[0], self.neg[1]]

    def time_getitem_numpy_int(self):
        self.M[self.npij[0], self.npij[1]]

    def time_resolver_int(self):
        IndexerResolver(self.M, self.ij)

    def time_resolver_numpy_int(self):
        IndexerResolver(self.M, self.npij)
