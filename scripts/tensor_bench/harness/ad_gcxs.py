"""pydata/sparse GCXS (default compressed axis), for the reduction and relayout rows only."""

import sparse
from ad_pydata import Adapter as _COOAdapter
from common import NA, parse_axes, parse_perm


class Adapter(_COOAdapter):
    name = "gcxs"

    def __init__(self, ds, coords, vals, shape):
        super().__init__(ds, coords, vals, shape)
        self.G = sparse.GCXS(self.X)
        del self.X

    def info(self):
        return {
            "version": f"sparse {sparse.__version__}",
            "compressed_axes": [int(a) for a in self.G.compressed_axes],
        }

    def extra_cases(self, group):
        return []

    def setup(self, group):
        if group not in {"reduce", "permute"}:
            raise NA("GCXS is benchmarked for reductions and permutations only")

    def case(self, group, case):
        G = self.G
        if group == "reduce":
            axes = parse_axes(case)
            if axes is not None:
                # GCXS reduces by re-compressing on the kept axes: indptr has prod(kept) + 1 entries
                kept = 1
                for a, s in enumerate(self.shape):
                    if a not in axes:
                        kept *= s
                if kept > 2**28:
                    raise NA(
                        f"re-compressing on the kept axes needs an indptr of {kept:.3g} entries"
                    )
            return lambda: G.sum(axis=axes)
        if group == "permute":
            p = parse_perm(case)
            return lambda: G.transpose(p)
        raise NA("GCXS is benchmarked for reductions and permutations only")
