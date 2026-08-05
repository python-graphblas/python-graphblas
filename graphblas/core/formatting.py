# The rich repr and _repr_html_ are hand-rendered here to reproduce pandas'
# DataFrame text/HTML output byte-for-byte without importing pandas. When pandas
# is installed we still read its display.* options so a user's option_context is
# honored; when it is absent we fall back to pandas' documented defaults.
import math
import re
import shutil

import numpy as np

from .. import backend, config, monoid, unary
from ..dtypes import BOOL
from ..exceptions import OutOfMemory
from .matrix import Matrix, TransposedMatrix
from .vector import Vector

try:
    import pandas as pd

    has_pandas = True
except ImportError:  # pragma: no cover (import)
    has_pandas = False

# pandas display.* defaults, used verbatim when pandas is not installed so the
# hand renderer produces the same output it would with a freshly imported pandas.
_DISPLAY_DEFAULTS = {
    "max_rows": 60,
    "min_rows": 10,
    "max_columns": 0,
    "width": 80,
    "expand_frame_repr": True,
    "precision": 6,
    "max_colwidth": 50,
    "chop_threshold": None,
    "colheader_justify": "right",
    "float_format": None,
    "html.border": 1,
    "html.use_mathjax": True,
}


def _display_option(name):
    """Read a pandas display.<name> option, or fall back to its default.

    Reading from pandas keeps a user's ``pd.option_context`` honored while the
    rendering logic itself stays pandas-free.
    """
    if has_pandas:
        return pd.get_option(f"display.{name}")
    return _DISPLAY_DEFAULTS[name]


# This was written by a complete novice at CSS.
# If you can help make it better, please do!
CSS_STYLE = """
<style>
table.gb-info-table {
    border: 1px solid black;
    max-width: 100%;
    margin-top: 0px;
    margin-bottom: 0px;
    padding-top: 0px;
    padding-bottom: 0px;
}

td.gb-info-name-cell {
    white-space: nowrap;
}

details.gb-arg-details {
    margin-top: 0px;
    margin-bottom: 0px;
    padding-top: 0px;
    padding-bottom: 5px;
    margin-left: 10px;
}

summary.gb-arg-summary {
    display: list-item;
    outline: none;
    margin-top: 0px;
    margin-bottom: 0px;
    padding-top: 0px;
    padding-bottom: 0px;
    margin-left: -10px;
}

details.gb-expr-details {
    margin-top: 0px;
    margin-bottom: 0px;
    padding-top: 0px;
    padding-bottom: 5px;
}

summary.gb-expr-summary {
    display: list-item;
    outline: none;
    margin-top: 0px;
    margin-bottom: 0px;
    padding-top: 0px;
    padding-bottom: 0px;
}

blockquote.gb-expr-blockquote {
    margin-top: 5px;
    margin-bottom: 0px;
    padding-top: 0px;
    padding-bottom: 0px;
    margin-left: 15px;
}

.gb-scalar {
    margin-top: 0px;
    margin-bottom: 0px;
    padding-top: 0px;
    padding-bottom: 5px;
}

/* modify pandas dataframe */
table.dataframe {
    margin-top: 0px;
    margin-bottom: 0px;
    padding-top: 0px;
    padding-bottom: 0px;
}

/* expression tooltips */
.expr-tooltip .tooltip-circle {
    background: #9a9cc6;
    color: #fff;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    padding-left: 4px;
    padding-right: 4px;
}
.expr-tooltip .tooltip-text {
    visibility: hidden;
    position: absolute;
    width: 450px;
    background: #eef;
    border: 1px solid #99a;
    text-align: left;
    border-radius: 6px;
    padding: 3px 3px 3px 8px;
    margin-left: 6px;
}
.expr-tooltip:hover .tooltip-text {
    visibility: visible;
}
.expr-tooltip code {
    background-color: #f8ffed;
}
</style>
"""


def _update_matrix_array(arr, matrix, rows, row_offset, columns, column_offset, *, mask=None):
    if rows is None and columns is None:
        if mask is None:
            submatrix = matrix
        else:
            submatrix = Matrix("UINT8", matrix._nrows, matrix._ncols, name="")
            if mask.structure:
                submatrix(matrix.S)[...] = 0 if mask.complement else 1
            else:
                submatrix(matrix.S)[...] = 1 if mask.complement else 0
                submatrix(matrix.V)[...] = 0 if mask.complement else 1
    else:
        if rows is None:
            rows = slice(None)
        if columns is None:
            columns = slice(None)
        if type(matrix) is TransposedMatrix:
            parent = matrix._matrix
            submatrix = Matrix(parent.dtype, parent._nrows, parent._ncols, name="")
            if parent._nvals > 0:  # pragma: no branch
                # Get val to support iso-valued matrices
                val = parent.reduce_scalar(monoid.any, allow_empty=False).new(name="")
                submatrix(parent.S)[columns, rows] = val
                submatrix(submatrix.S)[...] = parent
            if row_offset is not None or column_offset is not None:
                submatrix = submatrix[column_offset:, row_offset:].new(name="")
            submatrix = submatrix.T
        else:
            if mask is None:
                submatrix = Matrix(matrix.dtype, matrix._nrows, matrix._ncols, name="")
                # Get val to support iso-valued matrices
                if matrix._nvals > 0:
                    val = matrix.reduce_scalar(monoid.any).new(name="")
                    submatrix(matrix.S)[rows, columns] = val
                    submatrix(submatrix.S)[...] = matrix
            else:
                submatrix = Matrix("UINT8", matrix._nrows, matrix._ncols, name="")
                if mask.structure:
                    submatrix(matrix.S)[rows, columns] = 0 if mask.complement else 1
                else:
                    submatrix(matrix.S)[rows, columns] = 1 if mask.complement else 0
                    submatrix(matrix.V)[rows, columns] = 0 if mask.complement else 1
            if row_offset is not None or column_offset is not None:
                submatrix = submatrix[row_offset:, column_offset:].new(name="")
    rows, cols, vals = submatrix.to_coo()
    np_type = submatrix.dtype.np_type
    if submatrix.dtype._is_udt and np_type.subdtype is not None:
        vals = vals.tolist()
    if isinstance(vals, np.ndarray) and vals.dtype.names is not None:
        # Structured array: convert numpy.void elements to tuples for consistent display
        arr[rows, cols] = [tuple(v) for v in vals]
    else:
        arr[rows, cols] = vals
    if np.issubdtype(np_type, np.inexact):
        nulls = np.isnan(vals)
        arr[rows[nulls], cols[nulls]] = "nan"


def _update_vector_array(arr, vector, columns, column_offset, *, mask=None):
    if columns is None:
        if mask is None:
            subvector = vector
        else:
            subvector = Vector("UINT8", vector._size, name="")
            if mask.structure:
                subvector(vector.S)[...] = 0 if mask.complement else 1
            else:
                subvector(vector.S)[...] = 1 if mask.complement else 0
                subvector(vector.V)[...] = 0 if mask.complement else 1
    else:
        if mask is None:
            subvector = Vector(vector.dtype, vector._size, name="")
            # Get val to support iso-valued vectors
            if vector._nvals > 0:
                val = vector.reduce(monoid.any).new(name="")
                subvector(vector.S)[columns] = val
                subvector(subvector.S)[...] = vector
        else:
            subvector = Vector("UINT8", vector._size, name="")
            if mask.structure:
                subvector(vector.S)[columns] = 0 if mask.complement else 1
            else:
                subvector(vector.S)[columns] = 1 if mask.complement else 0
                subvector(vector.V)[columns] = 0 if mask.complement else 1
        if column_offset is not None:
            subvector = subvector[column_offset:].new(name="")
    cols, vals = subvector.to_coo()
    np_type = subvector.dtype.np_type
    if subvector.dtype._is_udt and np_type.subdtype is not None:
        vals = vals.tolist()
    if isinstance(vals, np.ndarray) and vals.dtype.names is not None:
        # Structured array: convert numpy.void elements to tuples for consistent display
        arr[0, cols] = [tuple(v) for v in vals]
    else:
        arr[0, cols] = vals
    if np.issubdtype(np_type, np.inexact):
        arr[0, cols[np.isnan(vals)]] = "nan"


def _get_max_columns():
    max_columns = _display_option("max_columns")
    if max_columns == 0:
        # We are probably in a terminal and pandas will automatically size the data correctly.
        # In this case, let's get a sufficiently large amount of data to show and defer to pandas.
        max_columns = 150
    return max_columns


def _get_chunk(length, min_length, max_length):
    if length <= max_length:
        chunk = list(range(length))
        chunk_groups = [(None, None)]
    else:
        half = min_length // 2
        first_chunk = list(range(half))
        second_chunk = list(range(length - half, length))
        chunk = list(range(max_length + 1))
        chunk[-half:] = second_chunk
        chunk_groups = [(first_chunk, None), (second_chunk, length - len(chunk))]
    return chunk, chunk_groups


class _Column:
    """One rendered column: a label, its raw cell values, and how to format them.

    ``kind`` selects the pandas array formatter this column would have used:
    "object" (GenericArrayFormatter), "float" (FloatArrayFormatter, also complex),
    or "int" (IntArrayFormatter). ``numeric`` mirrors pandas ``is_numeric_dtype``
    and controls whether the column header gets a leading space.
    """

    __slots__ = ("label", "values", "kind", "numeric")

    def __init__(self, label, values, kind, numeric):
        self.label = label
        self.values = values
        self.kind = kind
        self.numeric = numeric


class _GBFrame:
    """A minimal object/typed-column table, standing in for a pandas DataFrame.

    ``col_name`` is the columns' index name (pandas ``df.columns.name``); when set
    it appears in the top-left corner cell, as vector reprs rely on.
    """

    __slots__ = ("columns", "index", "col_name")

    def __init__(self, columns, index, col_name=None):
        self.columns = columns
        self.index = index
        self.col_name = col_name

    @property
    def ncols(self):
        return len(self.columns)

    @property
    def nrows(self):
        return len(self.index)

    def slice_cols(self, idx):
        return _GBFrame([self.columns[i] for i in idx], self.index, self.col_name)

    def slice_rows(self, idx):
        columns = [
            _Column(c.label, [c.values[i] for i in idx], c.kind, c.numeric) for c in self.columns
        ]
        return _GBFrame(columns, [self.index[i] for i in idx], self.col_name)


def _isna_cell(x):
    # Gaps in the dense grid are float NaN; present values (including displayed
    # "nan"/"inf") are never a bare float NaN, so this only flags the gaps.
    return x is None or (isinstance(x, float) and math.isnan(x))


def _count_present(arr):
    return sum(1 for x in arr.flat if not _isna_cell(x))


def _dtype_kind_numeric(dtype):
    kind = dtype.kind
    if kind in "fc":
        return "float", True
    if kind in "iu":
        return "int", True
    if kind == "b":
        # bool renders via the generic formatter, but is_numeric_dtype(bool) is True
        return "object", True
    return "object", False


def _make_dense_frame(arr, columns, index):
    nrows = len(index)
    out = []
    for j, label in enumerate(columns):
        vals = [("" if _isna_cell(arr[i, j]) else arr[i, j]) for i in range(nrows)]
        out.append(_Column(label, vals, "object", False))
    return _GBFrame(out, list(index))


def _make_coo_frame(label_arrays, add_dots):
    n = len(label_arrays[0][1])
    index = list(range(n))
    out = []
    for label, values in label_arrays:
        values = np.asarray(values)
        if add_dots:
            out.append(_Column(label, [*values.tolist(), "..."], "object", False))
        else:
            kind, numeric = _dtype_kind_numeric(values.dtype)
            out.append(_Column(label, values.tolist(), kind, numeric))
    if add_dots:
        index.append("...")
    return _GBFrame(out, index)


# --- cell formatting (reproduces pandas array formatters for object/int/float) ---

_NUMBER_RE = re.compile(r"^\s*[\+-]?[0-9]+\.[0-9]*$")


def _is_float_scalar(v):
    # Matches pandas.lib.is_float: python/numpy floats, but not bool/int/complex.
    return isinstance(v, (float, np.floating))


def _pprint(v):
    # Reproduces pandas printing.pprint_thing for our cell types (escape_chars for
    # tab/cr/nl, quote_strings=False): scalars -> str, sequences recurse.
    if isinstance(v, (list, tuple)):
        body = ", ".join(_pprint(e) for e in v)
        if isinstance(v, tuple) and len(v) == 1:
            body += ","
        return f"[{body}]" if isinstance(v, list) else f"({body})"
    s = str(v)
    return s.replace("\t", r"\t").replace("\r", r"\r").replace("\n", r"\n")


def _trim_zeros_single_float(s):
    s = s.rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


def _trim_zeros_float(str_floats):
    trimmed = list(str_floats)

    def is_number_with_decimal(x):
        return _NUMBER_RE.match(x) is not None

    def should_trim(values):
        numbers = [x for x in values if is_number_with_decimal(x)]
        return len(numbers) > 0 and all(x.endswith("0") for x in numbers)

    while should_trim(trimmed):
        trimmed = [x[:-1] if is_number_with_decimal(x) else x for x in trimmed]
    return [x + "0" if is_number_with_decimal(x) and x.endswith(".") else x for x in trimmed]


def _trim_zeros_complex(str_complexes):
    real_part, imag_part = [], []
    for x in str_complexes:
        trimmed = re.split(r"(?<!e)([j+-])", x)
        real_part.append("".join(trimmed[:-4]))
        imag_part.append("".join(trimmed[-4:-2]))
    n = len(str_complexes)
    padded_parts = _trim_zeros_float(real_part + imag_part)
    if len(padded_parts) == 0:
        return []
    padded_length = max(len(part) for part in padded_parts) - 1
    return [
        real_pt + imag_pt[0] + f"{imag_pt[1:]:>{padded_length}}" + "j"
        for real_pt, imag_pt in zip(padded_parts[:n], padded_parts[n:], strict=True)
    ]


def _value_formatter(fmt_str, threshold):
    def base(v):
        return fmt_str.format(value=v)

    if threshold is None:
        return base

    def formatter(v):
        try:
            mag = abs(v)
        except OverflowError:
            # abs() of a python complex overflows for finite components near
            # the float max, where numpy's abs returns inf. pandas computes the
            # magnitude with numpy, so such a value is never chopped.
            return base(v)
        # Chop to the value's own type of zero. pandas chops at the array
        # level, so a chopped complex renders " 0.000000+0.000000j"; a bare
        # float zero here would put a j-less string into the complex column,
        # which _trim_zeros_complex cannot parse.
        return base(v) if mag > threshold else base(type(v)(0))

    return formatter


def _format_reals_with_na(values, formatter, na_rep):
    return [na_rep if (v != v) else formatter(v) for v in values]


def _format_complex_with_na(values, formatter, na_rep):
    out = []
    for val in values:
        re_v, im_v = val.real, val.imag
        re_na, im_na = re_v != re_v, im_v != im_v
        if not re_na and not im_na:
            out.append(formatter(val))
        elif not re_na:
            out.append(f"{formatter(re_v)}+{na_rep}j")
        elif not im_na:
            imag_formatted = formatter(im_v).strip()
            if imag_formatted.startswith("-"):
                out.append(f"{na_rep}{imag_formatted}j")
            else:
                out.append(f"{na_rep}+{imag_formatted}j")
        else:
            out.append(f"{na_rep}+{na_rep}j")
    return out


def _format_float_column(values, digits):
    # Reproduces FloatArrayFormatter for fixed_width, leading_space=True, na_rep="NaN".
    arr = np.asarray(values)
    is_complex = np.iscomplexobj(arr)
    na_rep = "NaN"
    if (float_format := _display_option("float_format")) is not None:
        # A user display.float_format callable makes FloatArrayFormatter drop
        # fixed_width: each value (real or complex) is just float_format(value),
        # with no trailing-zero trim and no scientific switchover. Iterate the
        # numpy array (not .tolist()) so the callable receives numpy scalars, as
        # pandas does; e.g. a "%f"-style callable then casts complex the same way.
        return [na_rep if (v != v) else float_format(v) for v in arr]
    seq = arr.tolist()
    threshold = _display_option("chop_threshold")

    def format_with(fmt_str):
        formatter = _value_formatter(fmt_str, threshold)
        if is_complex:
            return _trim_zeros_complex(_format_complex_with_na(seq, formatter, na_rep))
        return _trim_zeros_float(_format_reals_with_na(seq, formatter, na_rep))

    result = format_with(f"{{value: .{digits:d}f}}")
    too_long = bool(result) and max(len(x) for x in result) > digits + 6
    abs_vals = np.abs(arr)
    has_large = bool((abs_vals > 1e6).any())
    has_small = bool(((abs_vals < 10.0 ** (-digits)) & (abs_vals > 0)).any())
    if has_small or (too_long and has_large):
        result = format_with(f"{{value: .{digits:d}e}}")
    return list(result)


def _justify(strings, width, mode="right"):
    if mode == "left":
        return [x.ljust(width) for x in strings]
    if mode == "center":
        return [x.center(width) for x in strings]
    return [x.rjust(width) for x in strings]


def _make_fixed_width(strings, justify="right", minimum=None):
    if not strings:
        return list(strings)
    max_len = max(len(x) for x in strings)
    if minimum is not None:
        max_len = max(minimum, max_len)
    conf_max = _display_option("max_colwidth")
    if conf_max is not None and max_len > conf_max:
        max_len = conf_max

    def just(x):
        if conf_max is not None and conf_max > 3 and len(x) > max_len:
            x = x[: max_len - 3] + "..."
        return x

    return _justify([just(x) for x in strings], max_len, justify)


def _format_labels(labels):
    # Reproduce pandas Index._format_flat(include_name=False) for our label types:
    # integer labels are padded to a uniform width (left-justified, with a sign
    # column when any are negative); string labels are left as-is.
    if labels and all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in labels):
        pattern = "{: d}" if any(x < 0 for x in labels) else "{:d}"
        strs = [pattern.format(x) for x in labels]
        width = max(len(s) for s in strs)
        return [s.ljust(width) for s in strs]
    return [str(x) for x in labels]


def _adjoin(space, lists):
    # Port of pandas printing.adjoin (ascii len/ljust); glues columns with `space`.
    lengths = [max(map(len, x)) + space for x in lists[:-1]]
    lengths.append(max(map(len, lists[-1])))
    max_len = max(map(len, lists))
    padded = []
    for i, lst in enumerate(lists):
        nl = [x.ljust(lengths[i]) for x in lst]
        nl = [" " * lengths[i]] * (max_len - len(lst)) + nl
        padded.append(nl)
    return "\n".join("".join(parts) for parts in zip(*padded, strict=True))


def _binify(cols, line_width):
    adjoin_width = 1
    bins = []
    curr_width = 0
    i_last = len(cols) - 1
    for i, w in enumerate(cols):
        w_adjoined = w + adjoin_width
        curr_width += w_adjoined
        if i_last == i:
            wrap = curr_width + 1 > line_width and i > 0
        else:
            wrap = curr_width + 2 > line_width and i > 0
        if wrap:
            bins.append(i)
            curr_width = w_adjoined
    bins.append(len(cols))
    return bins


def _console_width():
    # pandas repr sets the wrap width from console.get_console_size(); reuse it
    # when present so the wrap decision is identical. Without pandas (or if that
    # private module moved) fall back to display.width.
    if has_pandas:
        try:
            from pandas.io.formats.console import get_console_size

            return get_console_size()[0]
        except Exception:  # pragma: no cover (defensive across pandas versions)
            pass
    return _display_option("width")


class _TextFormatter:
    """Reproduces pandas DataFrameFormatter + StringFormatter for text repr."""

    def __init__(self, frame, max_rows, min_rows, max_cols):
        self.frame = frame
        self.max_rows = max_rows
        self.min_rows = min_rows
        self.max_cols = max_cols
        self.justify = _display_option("colheader_justify")
        self.tr_frame = frame
        self.tr_col_num = None
        self.tr_row_num = None
        self.max_cols_fitted = self._calc_max_cols_fitted()
        self.max_rows_fitted = self._calc_max_rows_fitted()
        self.truncate()

    def _is_in_terminal(self):
        return self.max_cols == 0 or self.max_rows == 0

    def _calc_max_cols_fitted(self):
        if not self._is_in_terminal():
            return self.max_cols
        width = shutil.get_terminal_size()[0]
        if self.max_cols == 0 and self.frame.ncols > width:
            return width
        return self.max_cols

    def _calc_max_rows_fitted(self):
        if self._is_in_terminal() and self.max_rows == 0:
            # rows available for data: terminal height minus dots + prompt + header
            return shutil.get_terminal_size()[1] - 3
        max_rows = self.max_rows
        if max_rows and self.frame.nrows > max_rows and self.min_rows:
            max_rows = min(self.min_rows, max_rows)
        return max_rows

    @property
    def is_truncated_horizontally(self):
        return bool(self.max_cols_fitted and self.frame.ncols > self.max_cols_fitted)

    @property
    def is_truncated_vertically(self):
        return bool(self.max_rows_fitted and self.frame.nrows > self.max_rows_fitted)

    @property
    def is_truncated(self):
        return self.is_truncated_horizontally or self.is_truncated_vertically

    def truncate(self):
        if self.is_truncated_horizontally:
            self._truncate_horizontally()
        if self.is_truncated_vertically:
            self._truncate_vertically()

    def _truncate_horizontally(self):
        col_num = self.max_cols_fitted // 2
        if col_num >= 1:
            _len = self.tr_frame.ncols
            self.tr_frame = self.tr_frame.slice_cols(
                [*range(col_num), *range(_len - col_num, _len)]
            )
        else:
            col_num = self.max_cols
            self.tr_frame = self.tr_frame.slice_cols(list(range(col_num)))
        self.tr_col_num = col_num

    def _truncate_vertically(self):
        row_num = self.max_rows_fitted // 2
        if row_num >= 1:
            _len = self.tr_frame.nrows
            self.tr_frame = self.tr_frame.slice_rows(
                [*range(row_num), *range(_len - row_num, _len)]
            )
        else:
            row_num = self.max_rows
            self.tr_frame = self.tr_frame.slice_rows(list(range(row_num)))
        self.tr_row_num = row_num

    def _format_col_raw(self, col):
        if col.kind == "int":
            return [f"{x: d}" for x in col.values]
        if col.kind == "float":
            return _format_float_column(col.values, _display_option("precision"))
        precision = _display_option("precision")
        float_format = _display_option("float_format")
        out = []
        for v in col.values:
            # A float NaN is excluded from pandas' float-format branch (it uses
            # is_float(v) & notna(v)) and rendered as the na_rep "NaN" instead.
            if _is_float_scalar(v) and not math.isnan(v):
                if float_format is not None:
                    # A user display.float_format callable replaces the default
                    # precision render (and adds no sign-space of its own).
                    out.append(float_format(v))
                else:
                    out.append(_trim_zeros_single_float(f"{v: .{precision}f}"))
            elif v is None:
                out.append(" None")
            elif _is_float_scalar(v):
                out.append(" NaN")
            else:
                out.append(f" {_pprint(v)}")
        return out

    def _get_body_strcols(self):
        # Column labels are formatted together (integer labels padded to a uniform
        # width) the way pandas Index._format_flat does, not per column.
        labels = _format_labels([col.label for col in self.tr_frame.columns])
        strcols = []
        for col, label in zip(self.tr_frame.columns, labels, strict=True):
            header = f" {label}" if col.numeric else label
            header_colwidth = len(header)
            # pandas fixes width twice: format_array right-justifies to the cell
            # content width, then the body pass re-justifies with colheader_justify
            # (which only matters when the header is wider, or when it is "left").
            fmt_values = _make_fixed_width(self._format_col_raw(col), "right")
            fmt_values = _make_fixed_width(fmt_values, self.justify, minimum=header_colwidth)
            max_len = max(max((len(x) for x in fmt_values), default=0), header_colwidth)
            cheader = _justify([header], max_len, self.justify)
            strcols.append(cheader + fmt_values)
        return strcols

    def _get_index_strcol(self):
        idx = _make_fixed_width([str(x) for x in self.tr_frame.index], justify="left")
        corner = "" if self.frame.col_name is None else str(self.frame.col_name)
        return [corner, *idx]

    def get_strcols(self):
        strcols = self._get_body_strcols()
        strcols.insert(0, self._get_index_strcol())
        return strcols

    @property
    def _adjusted_tr_col_num(self):
        return self.tr_col_num + 1  # index column is always shown

    def _insert_dot_separators(self, strcols):
        index_length = len(self._get_index_strcol())
        if self.is_truncated_horizontally:
            strcols.insert(self._adjusted_tr_col_num, [" ..."] * index_length)
        if self.is_truncated_vertically:
            self._insert_dots_vertical(strcols, index_length)
        return strcols

    def _insert_dots_vertical(self, strcols, index_length):
        n_header_rows = index_length - self.tr_frame.nrows
        row_num = self.tr_row_num
        for ix, col in enumerate(strcols):
            cwidth = len(col[row_num])
            is_dot_col = self.is_truncated_horizontally and ix == self._adjusted_tr_col_num
            dots = "..." if (cwidth > 3 or is_dot_col) else ".."
            if ix == 0:
                dot_mode = "left"
            elif is_dot_col:
                cwidth = 4
                dot_mode = "right"
            else:
                dot_mode = "right"
            col.insert(row_num + n_header_rows, _justify([dots], cwidth, dot_mode)[0])

    def _get_strcols(self):
        strcols = self.get_strcols()
        if self.is_truncated:
            strcols = self._insert_dot_separators(strcols)
        return strcols

    def _fit_to_terminal(self, strcols):
        lines = _adjoin(1, strcols).split("\n")
        max_len = max(len(x) for x in lines)
        width = shutil.get_terminal_size()[0]
        adj_dif = max_len - width + 1  # +1 to avoid too-wide repr (pandas GH #17023)
        col_lens = [max((len(x) for x in col), default=0) for col in strcols]
        n_cols = len(col_lens)
        while adj_dif > 0 and n_cols > 1:
            mid = round(n_cols / 2)
            adj_dif -= col_lens.pop(mid) + 1
            n_cols = len(col_lens)
        max_cols_fitted = max(n_cols - 1, 2)  # minus index column; show at least two
        self.max_cols_fitted = max_cols_fitted
        self.truncate()
        return _adjoin(1, self._get_strcols())

    def _join_multiline(self, strcols, line_width):
        adjoin_width = 1
        strcols = list(strcols)
        idx = strcols.pop(0)
        line_width -= max(len(x) for x in idx) + adjoin_width
        col_widths = [max((len(x) for x in col), default=0) for col in strcols]
        col_bins = _binify(col_widths, line_width)
        nbins = len(col_bins)
        blocks = []
        start = 0
        for i, end in enumerate(col_bins):
            row = strcols[start:end]
            row.insert(0, idx)
            if nbins > 1:
                nrows = len(row[-1])
                if end <= len(strcols) and i < nbins - 1:
                    row.append([" \\", *["  "] * (nrows - 1)])
                else:
                    row.append([" "] * nrows)
            blocks.append(_adjoin(adjoin_width, row))
            start = end
        return "\n\n".join(blocks)

    def to_string(self, line_width):
        strcols = self._get_strcols()
        if line_width is None:
            return _adjoin(1, strcols)
        if self.max_cols > 0:
            return self._join_multiline(strcols, line_width)
        return self._fit_to_terminal(strcols)


def _render_text(frame):
    max_cols = _display_option("max_columns")
    line_width = _console_width() if _display_option("expand_frame_repr") else None
    fmt = _TextFormatter(frame, _display_option("max_rows"), _display_option("min_rows"), max_cols)
    return fmt.to_string(line_width)


# The scoped style block pandas' NotebookFormatter emits ahead of the table.
_HTML_STYLE = (
    "<style scoped>\n"
    "    .dataframe tbody tr th:only-of-type {\n"
    "        vertical-align: middle;\n"
    "    }\n"
    "\n"
    "    .dataframe tbody tr th {\n"
    "        vertical-align: top;\n"
    "    }\n"
    "\n"
    "    .dataframe thead th {\n"
    "        text-align: right;\n"
    "    }\n"
    "</style>"
)


def _html_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class _HtmlBuilder:
    """Reproduces pandas NotebookFormatter (DataFrame._repr_html_) markup."""

    indent_delta = 2

    def __init__(self, fmt):
        self.fmt = fmt
        self.tr = fmt.tr_frame
        self.ncols = self.tr.ncols
        self.th = fmt.is_truncated_horizontally
        self.tv = fmt.is_truncated_vertically
        self.row_levels = 1  # single-level index, always shown
        self.elements = []

    def write(self, s, indent=0):
        self.elements.append(" " * indent + s)

    def _cell(self, s, kind, indent):
        rs = _html_escape(str(s)).strip().replace("  ", "&nbsp;&nbsp;")
        self.write(f"<{kind}>{rs}</{kind}>", indent)

    def write_tr(self, line, indent, header=False, align=None, nindex_levels=0):
        self.write("<tr>" if align is None else f'<tr style="text-align: {align};">', indent)
        inner = indent + self.indent_delta
        for i, s in enumerate(line):
            self._cell(s, "th" if (header or i < nindex_levels) else "td", inner)
        self.write("</tr>", indent)

    def _col_header(self, indent):
        row = ["" if self.tr.col_name is None else str(self.tr.col_name)]
        row.extend(_format_labels([col.label for col in self.tr.columns]))
        if self.th:
            row.insert(self.row_levels + self.fmt.tr_col_num, "...")
        self.write_tr(row, indent, header=True, align=self.fmt.justify)

    def _body(self, indent):
        index_labels = _format_labels(list(self.tr.index))
        col_cells = [
            _make_fixed_width(self.fmt._format_col_raw(col), "right") for col in self.tr.columns
        ]
        row = []
        for i in range(self.tr.nrows):
            if self.tv and i == self.fmt.tr_row_num:
                self.write_tr(["..."] * len(row), indent, nindex_levels=self.row_levels)
            row = [index_labels[i], *(col_cells[j][i] for j in range(self.ncols))]
            if self.th:
                row.insert(self.fmt.tr_col_num + self.row_levels, "...")
            self.write_tr(row, indent, nindex_levels=self.row_levels)

    def _table(self, indent=0):
        classes = "dataframe"
        if not _display_option("html.use_mathjax"):
            classes = "dataframe tex2jax_ignore mathjax_ignore"
        # pandas keeps the attribute for any non-None border, including 0.
        border = _display_option("html.border")
        border_attr = "" if border is None else f' border="{border}"'
        self.write(f'<table{border_attr} class="{classes}">', indent)
        self.write("<thead>", indent + self.indent_delta)
        self._col_header(indent + 2 * self.indent_delta)
        self.write("</thead>", indent + self.indent_delta)
        self.write("<tbody>", indent + self.indent_delta)
        self._body(indent + 2 * self.indent_delta)
        self.write("</tbody>", indent + self.indent_delta)
        self.write("</table>", indent)

    def render(self):
        self.write("<div>")
        self.write(_HTML_STYLE)
        self._table(0)
        self.write("</div>")
        return "\n".join(self.elements)


def _render_html(frame):
    fmt = _TextFormatter(
        frame,
        _display_option("max_rows"),
        _display_option("min_rows"),
        _display_option("max_columns"),
    )
    return _HtmlBuilder(fmt).render()


def _get_matrix_dataframe(matrix, max_rows, min_rows, max_columns, *, mask=None):
    if max_rows is None:  # pragma: no branch
        max_rows = _display_option("max_rows")
    if min_rows is None:  # pragma: no branch
        min_rows = _display_option("min_rows")
    if max_columns is None:  # pragma: no branch
        max_columns = _get_max_columns()
    rows, row_groups = _get_chunk(matrix._nrows, min_rows, max_rows)
    columns, column_groups = _get_chunk(matrix._ncols, max_columns, max_columns)
    arr = np.full((len(rows), len(columns)), np.nan, dtype=object)
    for row_group, row_offset in row_groups:
        for column_group, column_offset in column_groups:
            _update_matrix_array(
                arr,
                matrix,
                row_group,
                row_offset,
                column_group,
                column_offset,
                mask=mask,
            )
    present = _count_present(arr)
    truncated = (len(rows), len(columns)) != matrix.shape
    if (
        (mask is None or mask.structure)
        and truncated
        and min(matrix._nvals, max_rows if matrix._nvals <= max_rows else min_rows) > 2 * present
    ):
        # The data is sparse and it's better to show in COO format.
        # SS, SuiteSparse-specific: head
        num_rows = matrix._nvals if matrix._nvals <= max_rows else min_rows
        if matrix._is_transposed:
            cols, rows, vals = matrix._matrix.ss.head(num_rows, sort=True)
        else:
            rows, cols, vals = matrix.ss.head(num_rows, sort=True)
        if mask is not None:
            if mask.complement:
                vals = np.zeros(vals.size, dtype=np.uint8)
            else:
                vals = np.ones(vals.size, dtype=np.uint8)
        return _make_coo_frame(
            [("row", rows), ("col", cols), ("val", vals)], num_rows < matrix._nvals
        )
    if mask is not None and not mask.structure and truncated:
        # This performs more calculation and uses more memory than I would prefer.
        # Perhaps we could use the efficient "constant vector or matrix" trick.
        nonzero = matrix.apply(unary.one["UINT8"]).new(mask=matrix.V, name="")
        num_rows = matrix._nvals if matrix._nvals <= max_rows else min_rows
        if min(nonzero._nvals, num_rows) > 2 * present:
            rows, cols, vals = nonzero.ss.head(num_rows, sort=True)
            if mask.complement:
                if not vals.flags.writeable:  # pragma: no cover (safety)
                    vals = vals.copy()
                vals[:] = 0
            return _make_coo_frame(
                [("row", rows), ("col", cols), ("val", vals)], num_rows < nonzero._nvals
            )
    return _make_dense_frame(arr, columns, rows)


def _get_vector_dataframe(vector, max_rows, min_rows, max_columns, *, mask=None):
    if max_rows is None:  # pragma: no branch
        max_rows = _display_option("max_rows")
    if min_rows is None:  # pragma: no branch
        min_rows = _display_option("min_rows")
    if max_columns is None:  # pragma: no branch
        max_columns = _get_max_columns()
    columns, column_groups = _get_chunk(vector._size, max_columns, max_columns)
    arr = np.full((1, len(columns)), np.nan, dtype=object)
    for column_group, column_offset in column_groups:
        _update_vector_array(arr, vector, column_group, column_offset, mask=mask)
    present = _count_present(arr)
    truncated = len(columns) != vector._size
    if (
        (mask is None or mask.structure)
        and truncated
        and min(vector._nvals, max_rows if vector._nvals <= max_rows else min_rows) > 2 * present
    ):
        # The data is sparse and it's better to show in COO format.
        # SS, SuiteSparse-specific: head
        num_rows = vector._nvals if vector._nvals <= max_rows else min_rows
        indices, vals = vector.ss.head(num_rows, sort=True)
        if mask is not None:
            if mask.complement:
                vals = np.zeros(vals.size, dtype=np.uint8)
            else:
                vals = np.ones(vals.size, dtype=np.uint8)
        return _make_coo_frame([("index", indices), ("val", vals)], num_rows < vector._nvals)
    if mask is not None and not mask.structure and truncated:
        # This performs more calculation and uses more memory than I would prefer.
        # Perhaps we could use the efficient "constant vector or matrix" trick.
        nonzero = vector.apply(unary.one["UINT8"]).new(mask=vector.V, name="")
        num_rows = vector._nvals if vector._nvals <= max_rows else min_rows
        if min(nonzero._nvals, num_rows) > 2 * present:
            indices, vals = nonzero.ss.head(num_rows, sort=True)
            if mask.complement:
                if not vals.flags.writeable:  # pragma: no cover (safety)
                    vals = vals.copy()
                vals[:] = 0
            return _make_coo_frame([("index", indices), ("val", vals)], num_rows < nonzero._nvals)
    return _make_dense_frame(arr, columns, [""])


def get_format(x, is_transposed=False):
    # SS, SuiteSparse-specific: format (ends with "r" or "c"), and is_iso
    fmt = x.ss.format
    if is_transposed:
        fmt = fmt[:-1] + ("c" if fmt[-1] == "r" else "r")
    if x.ss.is_iso:
        return f"{fmt} (iso)"
    return fmt


def matrix_info(matrix, *, mask=None, expr=None, for_html=True):
    if mask is not None:
        if for_html:
            name = f"{type(mask).__name__}\nof\ngb.{type(matrix).__name__}"
        else:
            name = [f"{type(mask).__name__}", f"of gb.{type(matrix).__name__}"]
    else:
        name = f"gb.{type(matrix).__name__}"
    keys = ["nvals", "nrows", "ncols", "dtype"]
    vals = [matrix._nvals, matrix._nrows, matrix._ncols, matrix.dtype.name]
    if expr is None and backend == "suitesparse":
        keys.append("format")
        if type(matrix) is Matrix:
            vals.append(get_format(matrix))
        else:  # TransposedMatrix
            vals.append(get_format(matrix._matrix, is_transposed=True))
    return name, keys, vals


def vector_info(vector, *, mask=None, expr=None, for_html=True):
    if mask is not None:
        if for_html:
            name = f"{type(mask).__name__}\nof\ngb.{type(vector).__name__}"
        else:
            name = [f"{type(mask).__name__}", f"of gb.{type(vector).__name__}"]
    else:
        name = f"gb.{type(vector).__name__}"
    keys = ["nvals", "size", "dtype"]
    vals = [vector._nvals, vector._size, vector.dtype.name]
    if expr is None and backend == "suitesparse":
        keys.append("format")
        vals.append(get_format(vector))
    return name, keys, vals


def create_header_html(name, keys, vals):
    text = [
        '<div>\n<table class="gb-info-table">\n'
        "  <tr>\n"
        f'    <td rowspan="2" class="gb-info-name-cell"><pre>{name}</pre></td>\n'
    ]
    text.extend(f"    <td><pre>{key}</pre></td>\n" for key in keys)
    text.append("  </tr>\n  <tr>\n")
    text.extend(f"    <td>{val}</td>\n" for val in vals)
    text.append("  </tr>\n</table>\n</div>\n")
    return "".join(text)


def matrix_header_html(matrix, *, mask=None):
    name, keys, vals = matrix_info(matrix, mask=mask, for_html=True)
    return create_header_html(name, keys, vals)


def matrix_expression_header_html(matrix, expr):
    _, keys, vals = matrix_info(matrix, expr=expr, for_html=True)
    name = expr._format_expr_html()
    return create_header_html(name, keys, vals)


def vector_header_html(vector, *, mask=None):
    name, keys, vals = vector_info(vector, mask=mask, for_html=True)
    return create_header_html(name, keys, vals)


def vector_expression_header_html(matrix, expr):
    _, keys, vals = vector_info(matrix, expr=expr, for_html=True)
    name = expr._format_expr_html()
    return create_header_html(name, keys, vals)


def _format_html(name, header, frame, collapse):
    state = "" if collapse else " open"
    details = _render_html(frame)
    return (
        "<div>"
        f"{CSS_STYLE}"
        f'<details{state} class="gb-arg-details">'
        '<summary class="gb-arg-summary">'
        f"<tt>{name}</tt>{header}"
        "</summary>"
        f"{details}"
        "</details>"
        "</div>"
    )


def format_matrix_html(
    matrix,
    *,
    max_rows=None,
    min_rows=None,
    max_columns=None,
    mask=None,
    collapse=False,
    expr=None,
):
    if expr is not None:
        header = matrix_expression_header_html(matrix, expr)
        name = "__EXPR__"
    else:
        header = matrix_header_html(matrix, mask=mask)
        name = (matrix if mask is None else mask)._name_html
    df = _get_matrix_dataframe(matrix, max_rows, min_rows, max_columns, mask=mask)
    return _format_html(name, header, df, collapse)


def format_vector_html(
    vector,
    *,
    max_rows=None,
    min_rows=None,
    max_columns=None,
    mask=None,
    collapse=False,
    expr=None,
):
    if expr is not None:
        header = vector_expression_header_html(vector, expr)
        name = "__EXPR__"
    else:
        header = vector_header_html(vector, mask=mask)
        name = (vector if mask is None else mask)._name_html
    df = _get_vector_dataframe(vector, max_rows, min_rows, max_columns, mask=mask)
    return _format_html(name, header, df, collapse)


def format_scalar_html(scalar, expr=None):
    top_name = scalar._name_html if expr is None else "__EXPR__"
    box_name = "gb.Scalar" if expr is None else expr._format_expr_html()
    header = create_header_html(box_name, ["value", "dtype"], [scalar.value, scalar.dtype])
    return f'{CSS_STYLE}<div class="gb-scalar"><tt>{top_name}</tt>{header}</div>'


def format_scalar(scalar, expr=None):
    return create_header(
        "gb.Scalar",
        ["value", "dtype"],
        [scalar.value, scalar.dtype],
        name=scalar.name,
    )


def get_expr_result(expr, html=False):
    try:
        val = expr.new()
    except OutOfMemory:
        arg_string = "Result is too large to compute!"
        if html:
            arg_string = f'<span style="color: red">{arg_string}</span>'
    else:
        name = val.name
        val.name = "Result"
        if html:
            arg_string = f"{val._repr_html_(expr=expr)}"
        else:
            arg_string = val.__repr__(expr=expr)
        val.name = name
    return arg_string


def _format_expression(expr, header):
    topline = (
        f"<tt><b>gb.{type(expr).__name__}</b></tt>"
        '&nbsp;<span class="expr-tooltip">'
        '<span class="tooltip-circle">?</span>'
        '<span class="tooltip-text"><em>'
        "Do <code>expr.new()</code> or <code>other << expr</code> to calculate the expression."
        "</em></span></span>"
    )

    computed = ""
    if config.get("autocompute"):
        computed = get_expr_result(expr, html=True)
        if "__EXPR__" in computed:
            return computed.replace("<tt>__EXPR__</tt>", topline)

    return (
        "<div>"
        f"{CSS_STYLE}"
        '<details class="gb-expr-details">'
        '<summary class="gb-expr-summary">'
        f"{topline}"
        f"{header}"
        "</summary>"
        f"{computed}"
        "</details>"
        "</div>"
    )


def format_matrix_expression_html(expr):
    expr_html = expr._format_expr_html()
    header = create_header_html(
        expr_html, ["nrows", "ncols", "dtype"], [expr._nrows, expr._ncols, expr.dtype]
    )
    return _format_expression(expr, header)


def get_result_string(expr):
    if config.get("autocompute"):
        arg_string = get_expr_result(expr)
        arg_string += "\n\n"
    else:
        arg_string = ""
    return arg_string


def format_matrix_expression(expr):
    expr_repr = expr._format_expr()
    name = f"gb.{type(expr).__name__}"
    header = create_header(
        expr_repr,
        ["nrows", "ncols", "dtype"],
        [expr._nrows, expr._ncols, expr.dtype],
        name=name,
        quote=False,
    )
    arg_string = get_result_string(expr)
    return (
        f"{header}\n\n"
        f"{arg_string}"
        "Do expr.new() or other << expr to calculate the expression."
    )


def format_vector_expression_html(expr):
    expr_html = expr._format_expr_html()
    header = create_header_html(expr_html, ["size", "dtype"], [expr._size, expr.dtype])
    return _format_expression(expr, header)


def format_vector_expression(expr):
    expr_repr = expr._format_expr()
    name = f"gb.{type(expr).__name__}"
    header = create_header(
        expr_repr, ["size", "dtype"], [expr._size, expr.dtype], name=name, quote=False
    )
    arg_string = get_result_string(expr)
    return (
        f"{header}\n\n"
        f"{arg_string}"
        "Do expr.new() or other << expr to calculate the expression."
    )


def format_scalar_expression_html(expr):
    expr_html = expr._format_expr_html()
    header = create_header_html(expr_html, ["dtype"], [expr.dtype])
    return _format_expression(expr, header)


def format_scalar_expression(expr):
    expr_repr = expr._format_expr()
    name = f"gb.{type(expr).__name__}"
    header = create_header(expr_repr, ["dtype"], [expr.dtype], name=name, quote=False)
    arg_string = get_result_string(expr)
    return (
        f"{header}\n\n"
        f"{arg_string}"
        "Do expr.new() or other << expr to calculate the expression."
    )


def create_header(type_name, keys, vals, *, lower_border=False, name="", quote=True):
    vals = [str(x) for x in vals]
    if name and quote:
        name = f'"{name}"'
    key_text = []
    val_text = []
    for key, val in zip(keys, vals, strict=True):
        width = max(len(key), len(val)) + 2
        key_text.append(key.rjust(width))
        val_text.append(val.rjust(width))
    if isinstance(type_name, str):
        name_width = max(len(type_name), len(name))
        lines = [
            f"{name.ljust(name_width)}{''.join(key_text)}",
            f"{type_name.ljust(name_width)}{''.join(val_text)}",
        ]
    else:
        name_width = max(map(len, type_name))
        name_width = max(name_width, len(name))
        lines = [f"{name.ljust(name_width)}{''.join(key_text)}"]
        lines.extend(line.ljust(name_width) for line in type_name)
        lines[-1] += "".join(val_text)
    if lower_border:
        lines.append("-" * len(lines[0]))
    return "\n".join(lines)


def format_matrix(matrix, *, max_rows=None, min_rows=None, max_columns=None, mask=None, expr=None):
    name, keys, vals = matrix_info(matrix, mask=mask, expr=expr, for_html=False)
    header = create_header(
        name,
        keys,
        vals,
        lower_border=True,
        name=matrix.name if mask is None else mask.name,
    )
    if 0 not in matrix.shape:
        frame = _get_matrix_dataframe(matrix, max_rows, min_rows, max_columns, mask=mask)
        return f"{header}\n{_render_text(frame)}"
    return header


def format_vector(vector, *, max_rows=None, min_rows=None, max_columns=None, mask=None, expr=None):
    name, keys, vals = vector_info(vector, mask=mask, expr=expr, for_html=False)
    header = create_header(
        name,
        keys,
        vals,
        lower_border=True,
        name=vector.name if mask is None else mask.name,
    )
    if vector._size > 0:
        frame = _get_vector_dataframe(vector, max_rows, min_rows, max_columns, mask=mask)
        if frame.columns[0].label != "index":
            # Dense vectors label the corner "index" and the single row "value".
            frame.col_name = "index"
            frame.index = ["value"]
        return f"{header}\n{_render_text(frame)}"
    return header


def _format_infix_expression(expr, header, expr_name):
    topline = (
        f"<tt><b>gb.{type(expr).__name__}</b></tt>"
        '&nbsp;<span class="expr-tooltip">'
        '<span class="tooltip-circle">?</span>'
        '<span class="tooltip-text"><em>'
        f"Do <code>op(expr)</code> to create a <tt>{expr.output_type.__name__}</tt>"
        f" for <tt>{expr.method_name}</tt>."
        f"<br>For example: <code>{expr._example_op}({expr_name})</code>"
        "</em></span></span>"
    )

    computed = ""
    if config.get("autocompute") and (
        expr.method_name not in {"ewise_add", "ewise_mult"}
        or expr.left.dtype == BOOL
        and expr.right.dtype == BOOL
    ):
        computed = get_expr_result(expr, html=True)
        if "__EXPR__" in computed:
            return computed.replace("<tt>__EXPR__</tt>", topline)

    return (
        "<div>"
        f"{CSS_STYLE}"
        '<details class="gb-expr-details">'
        '<summary class="gb-expr-summary">'
        f"{topline}"
        f"{header}"
        "</summary>"
        f"{computed}"
        "</details>"
        "</div>"
    )


def format_scalar_infix_expression(expr):
    expr_repr = expr._format_expr()
    name = f"gb.{type(expr).__name__}"
    header = create_header(
        expr_repr,
        ["left_dtype", "right_dtype"],
        [expr.left.dtype, expr.right.dtype],
        name=name,
        quote=False,
    )
    arg_string = get_result_string(expr)
    return (
        f"{header}\n\n"
        f"{arg_string}"
        f"Do op(expr) to create a {expr.output_type.__name__} for {expr.method_name}.\n"
        f"For example: {expr._example_op}({expr_repr})"
    )


def format_scalar_infix_expression_html(expr):
    expr_html = expr._format_expr_html()
    header = create_header_html(
        expr_html,
        ["left_dtype", "right_dtype"],
        [expr.left.dtype, expr.right.dtype],
    )
    return _format_infix_expression(expr, header, expr_html)


def get_infix_result_string(expr):
    if (
        expr.method_name not in {"ewise_add", "ewise_mult"}
        or expr.left.dtype == BOOL
        and expr.right.dtype == BOOL
    ):
        arg_string = get_result_string(expr)
    else:
        arg_string = ""
    return arg_string


def format_vector_infix_expression(expr):
    expr_repr = expr._format_expr()
    name = f"gb.{type(expr).__name__}"
    header = create_header(
        expr_repr,
        ["size", "left_dtype", "right_dtype"],
        [expr._size, expr.left.dtype, expr.right.dtype],
        name=name,
        quote=False,
    )
    arg_string = get_infix_result_string(expr)
    return (
        f"{header}\n\n"
        f"{arg_string}"
        f"Do op(expr) to create a {expr.output_type.__name__} for {expr.method_name}.\n"
        f"For example: {expr._example_op}({expr_repr})"
    )


def format_vector_infix_expression_html(expr):
    expr_html = expr._format_expr_html()
    header = create_header_html(
        expr_html,
        ["size", "left_dtype", "right_dtype"],
        [expr._size, expr.left.dtype, expr.right.dtype],
    )
    return _format_infix_expression(expr, header, expr_html)


def format_matrix_infix_expression(expr):
    expr_repr = expr._format_expr()
    name = f"gb.{type(expr).__name__}"
    header = create_header(
        expr_repr,
        ["nrows", "ncols", "left_dtype", "right_dtype"],
        [expr._nrows, expr._ncols, expr.left.dtype, expr.right.dtype],
        name=name,
        quote=False,
    )
    arg_string = get_infix_result_string(expr)
    return (
        f"{header}\n\n"
        f"{arg_string}"
        f"Do op(expr) to create a {expr.output_type.__name__} for {expr.method_name}.\n"
        f"For example: {expr._example_op}({expr_repr})"
    )


def format_matrix_infix_expression_html(expr):
    expr_html = expr._format_expr_html()
    header = create_header_html(
        expr_html,
        ["nrows", "ncols", "right_dtype", "left_dtype"],
        [expr._nrows, expr._ncols, expr.left.dtype, expr.right.dtype],
    )
    return _format_infix_expression(expr, header, expr_html)


def format_index_expression(expr):
    name = f"gb.{type(expr).__name__}"
    expr_repr = expr._format_expr()
    keys = []
    values = []
    if expr.output_type is Vector:
        keys.append("size")
        values.append(expr._size)
    elif expr.output_type is Matrix:
        keys.extend(["nrows", "ncols"])
        values.extend([expr._nrows, expr._ncols])
    keys.append("dtype")
    values.append(expr.dtype)
    header = create_header(
        expr_repr,
        keys,
        values,
        name=name,
        quote=False,
    )
    arg_string = get_result_string(expr)
    c = expr.output_type.__name__[0]
    return (
        f"{header}\n\n"
        f"{arg_string}"
        f"This expression may be used to extract or assign a {expr.output_type.__name__}.\n"
        f"Example extract: {expr_repr}.new()\n"
        f"Example assign: {expr_repr} << {'M' if c == 'M' else c.lower()}"
    )


def format_index_expression_html(expr):
    expr_repr = expr._format_expr()
    c = expr.output_type.__name__[0]
    c = "M" if c == "M" else c.lower()
    topline = (
        f"<tt><b>gb.{type(expr).__name__}</b></tt>"
        '&nbsp;<span class="expr-tooltip">'
        '<span class="tooltip-circle">?</span>'
        '<span class="tooltip-text"><em>'
        f"This expression may be used to extract or assign a <tt>{expr.output_type.__name__}</tt>."
        f"<br>Example extract: <code>{expr_repr}.new()</code>"
        f"<br>Example assign: <code>{expr_repr} << {'M' if c == 'M' else c.lower()}</code>"
        "</em></span></span>"
    )

    computed = ""
    if config.get("autocompute"):
        computed = get_expr_result(expr, html=True)
        if "__EXPR__" in computed:
            return computed.replace("<tt>__EXPR__</tt>", topline)
        # BRANCH NOT COVERED

    keys = []
    values = []
    if expr.output_type is Vector:
        keys.append("size")
        values.append(expr._size)
    elif expr.output_type is Matrix:
        keys.extend(["nrows", "ncols"])
        values.extend([expr._nrows, expr._ncols])
    keys.append("dtype")
    values.append(expr.dtype)
    header = create_header_html(
        expr_repr,
        keys,
        values,
    )
    return (
        "<div>"
        f"{CSS_STYLE}"
        '<details class="gb-expr-details">'
        '<summary class="gb-expr-summary">'
        f"{topline}"
        f"{header}"
        "</summary>"
        f"{computed}"
        "</details>"
        "</div>"
    )
