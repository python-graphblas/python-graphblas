"""repr / _repr_html_ rendering.

A repr formats a table of the values and, for a large object, decides which rows and
columns to leave out. Small reprs measure the overhead; large reprs check that leaving
values out stays cheap (it must not format each of about 1e6 nonzeros).
"""

from graphblas import Matrix, Vector

try:
    from . import common
except ImportError:
    import common


class ReprSmall:
    def setup(self):
        self.M = Matrix.from_coo([0, 1, 2], [0, 1, 2], [1.0, 2.0, 3.0], nrows=4, ncols=4)
        self.v = Vector.from_coo([0, 2], [1.0, 2.0], size=5)

    def time_repr_matrix(self):
        repr(self.M)

    def time_repr_vector(self):
        repr(self.v)

    def time_repr_html_matrix(self):
        self.M._repr_html_()

    def time_repr_html_vector(self):
        self.v._repr_html_()


class ReprLarge:
    # Should be bounded by truncation, but give it room in case a regression makes
    # it render the whole object.
    timeout = 120

    def setup(self):
        self.M = common.make_matrix(seed=50)
        self.v = common.make_vector(seed=51)

    def time_repr_matrix(self):
        repr(self.M)

    def time_repr_vector(self):
        repr(self.v)

    def time_repr_html_matrix(self):
        self.M._repr_html_()
