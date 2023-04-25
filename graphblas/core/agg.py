"""graphblas.core.agg namespace is deprecated; please use graphblas.core.operator.agg instead.

.. deprecated:: 2023.3.0
`graphblas.core.agg` will be removed in a future release.
Use `graphblas.core.operator.agg` instead.
Will be removed in version 2023.11.0 or later.

"""
import warnings

from .operator.agg import *  # pylint: disable=wildcard-import,unused-wildcard-import

warnings.warn(
    "graphblas.core.agg namespace is deprecated; please use graphblas.core.operator.agg instead.",
    DeprecationWarning,
    stacklevel=1,
)
