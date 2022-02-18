import numpy as np

from . import config, monoid, unary
from .dtypes import BOOL
from .exceptions import OutOfMemory
from .matrix import Matrix, TransposedMatrix
from .vector import Vector

try:
    import pandas as pd

    has_pandas = True
except ImportError:  # pragma: no cover
    has_pandas = False

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
    line-height: 100%;
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
</style>
"""


def _update_matrix_dataframe(df, matrix, rows, row_offset, columns, column_offset, *, mask=None):
    if rows is None and columns is None:
        if mask is None:
            submatrix = matrix
        else:
            submatrix = Matrix.new("UINT8", matrix._nrows, matrix._ncols, name="")
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
            submatrix = Matrix.new(parent.dtype, parent._nrows, parent._ncols, name="")
            if parent._nvals > 0:
                # Get val to support iso-valued matrices
                val = parent.reduce_scalar(monoid.any, allow_empty=False).new(name="")
                submatrix(parent.S)[columns, rows] = val
                submatrix(submatrix.S)[...] = parent
            if row_offset is not None or column_offset is not None:
                submatrix = submatrix[column_offset:, row_offset:].new(name="")
            submatrix = submatrix.T
        else:
            if mask is None:
                submatrix = Matrix.new(matrix.dtype, matrix._nrows, matrix._ncols, name="")
                # Get val to support iso-valued matrices
                if matrix._nvals > 0:
                    val = matrix.reduce_scalar(monoid.any).new(name="")
                    submatrix(matrix.S)[rows, columns] = val
                    submatrix(submatrix.S)[...] = matrix
            else:
                submatrix = Matrix.new("UINT8", matrix._nrows, matrix._ncols, name="")
                if mask.structure:
                    submatrix(matrix.S)[rows, columns] = 0 if mask.complement else 1
                else:
                    submatrix(matrix.S)[rows, columns] = 1 if mask.complement else 0
                    submatrix(matrix.V)[rows, columns] = 0 if mask.complement else 1
            if row_offset is not None or column_offset is not None:
                submatrix = submatrix[row_offset:, column_offset:].new(name="")
    rows, cols, vals = submatrix.to_values()
    df.values[rows, cols] = vals
    nulls = np.isnan(vals)
    df.values[rows[nulls], cols[nulls]] = "nan"


def _update_vector_dataframe(df, vector, columns, column_offset, *, mask=None):
    if columns is None:
        if mask is None:
            subvector = vector
        else:
            subvector = Vector.new("UINT8", vector._size, name="")
            if mask.structure:
                subvector(vector.S)[...] = 0 if mask.complement else 1
            else:
                subvector(vector.S)[...] = 1 if mask.complement else 0
                subvector(vector.V)[...] = 0 if mask.complement else 1
    else:
        if mask is None:
            subvector = Vector.new(vector.dtype, vector._size, name="")
            # Get val to support iso-valued vectors
            if vector._nvals > 0:
                val = vector.reduce(monoid.any).new(name="")
                subvector(vector.S)[columns] = val
                subvector(subvector.S)[...] = vector
        else:
            subvector = Vector.new("UINT8", vector._size, name="")
            if mask.structure:
                subvector(vector.S)[columns] = 0 if mask.complement else 1
            else:
                subvector(vector.S)[columns] = 1 if mask.complement else 0
                subvector(vector.V)[columns] = 0 if mask.complement else 1
        if column_offset is not None:
            subvector = subvector[column_offset:].new(name="")
    cols, vals = subvector.to_values()
    df.values[0, cols] = vals
    df.values[0, cols[np.isnan(vals)]] = "nan"


def _get_max_columns():
    max_columns = pd.options.display.max_columns
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


def _get_matrix_dataframe(matrix, max_rows, min_rows, max_columns, *, mask=None):
    if not has_pandas:
        return
    if max_rows is None:  # pragma: no branch
        max_rows = pd.options.display.max_rows
    if min_rows is None:  # pragma: no branch
        min_rows = pd.options.display.min_rows
    if max_columns is None:  # pragma: no branch
        max_columns = _get_max_columns()
    rows, row_groups = _get_chunk(matrix._nrows, min_rows, max_rows)
    columns, column_groups = _get_chunk(matrix._ncols, max_columns, max_columns)
    df = pd.DataFrame(columns=columns, index=rows)
    for row_group, row_offset in row_groups:
        for column_group, column_offset in column_groups:
            _update_matrix_dataframe(
                df,
                matrix,
                row_group,
                row_offset,
                column_group,
                column_offset,
                mask=mask,
            )
    if (
        (mask is None or mask.structure)
        and df.shape != matrix.shape
        and min(matrix._nvals, max_rows if matrix._nvals <= max_rows else min_rows)
        > 2 * df.count().sum()
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
        df = pd.DataFrame({"row": rows, "col": cols, "val": vals})
        if num_rows < matrix._nvals:
            df.loc["..."] = ["..."] * 3
        return df
    if mask is not None and not mask.structure and df.shape != matrix.shape:
        # This performs more calculation and uses more memory than I would prefer.
        # Perhaps we could use the efficient "constant vector or matrix" trick.
        nonzero = matrix.apply(unary.one["UINT8"]).new(mask=matrix.V, name="")
        num_rows = matrix._nvals if matrix._nvals <= max_rows else min_rows
        if min(nonzero._nvals, num_rows) > 2 * df.count().sum():
            rows, cols, vals = nonzero.ss.head(num_rows, sort=True)
            if mask.complement:
                if not vals.flags.writeable:
                    vals = vals.copy()
                vals[:] = 0
            df = pd.DataFrame({"row": rows, "col": cols, "val": vals})
            if num_rows < nonzero._nvals:
                df.loc["..."] = ["..."] * 3
            return df
    return df.where(pd.notnull(df), "")


def _get_vector_dataframe(vector, max_rows, min_rows, max_columns, *, mask=None):
    if not has_pandas:
        return
    if max_rows is None:  # pragma: no branch
        max_rows = pd.options.display.max_rows
    if min_rows is None:  # pragma: no branch
        min_rows = pd.options.display.min_rows
    if max_columns is None:  # pragma: no branch
        max_columns = _get_max_columns()
    columns, column_groups = _get_chunk(vector._size, max_columns, max_columns)
    df = pd.DataFrame(columns=columns, index=[""])
    for column_group, column_offset in column_groups:
        _update_vector_dataframe(df, vector, column_group, column_offset, mask=mask)
    if (
        (mask is None or mask.structure)
        and df.size != vector._size
        and min(vector._nvals, max_rows if vector._nvals <= max_rows else min_rows)
        > 2 * df.count().sum()
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
        df = pd.DataFrame({"index": indices, "val": vals})
        if num_rows < vector._nvals:
            df.loc["..."] = ["..."] * 2
        return df
    if mask is not None and not mask.structure and df.size != vector._size:
        # This performs more calculation and uses more memory than I would prefer.
        # Perhaps we could use the efficient "constant vector or matrix" trick.
        nonzero = vector.apply(unary.one["UINT8"]).new(mask=vector.V, name="")
        num_rows = vector._nvals if vector._nvals <= max_rows else min_rows
        if min(nonzero._nvals, num_rows) > 2 * df.count().sum():
            indices, vals = nonzero.ss.head(num_rows, sort=True)
            if mask.complement:
                if not vals.flags.writeable:
                    vals = vals.copy()
                vals[:] = 0
            df = pd.DataFrame({"index": indices, "val": vals})
            if num_rows < nonzero._nvals:
                df.loc["..."] = ["..."] * 2
            return df
    return df.where(pd.notnull(df), "")


def get_format(x, is_transposed=False):
    # SS, SuiteSparse-specific: format (ends with "r" or "c"), and is_iso
    fmt = x.ss.format
    if is_transposed:
        fmt = fmt[:-1] + ("c" if fmt[-1] == "r" else "r")
    if x.ss.is_iso:
        return f"{fmt} (iso)"
    return fmt


def matrix_info(matrix, *, mask=None, for_html=True):
    if mask is not None:
        if for_html:
            name = f"{type(mask).__name__}\nof\ngrblas.{type(matrix).__name__}"
        else:
            name = [f"{type(mask).__name__}", f"of grblas.{type(matrix).__name__}"]
    else:
        name = f"grblas.{type(matrix).__name__}"
    keys = ["nvals", "nrows", "ncols", "dtype", "format"]
    vals = [matrix._nvals, matrix._nrows, matrix._ncols, matrix.dtype.name]
    if type(matrix) is Matrix:
        vals.append(get_format(matrix))
    else:  # TransposedMatrix
        vals.append(get_format(matrix._matrix, is_transposed=True))
    return name, keys, vals


def vector_info(vector, *, mask=None, for_html=True):
    if mask is not None:
        if for_html:
            name = f"{type(mask).__name__}\nof\ngrblas.{type(vector).__name__}"
        else:
            name = [f"{type(mask).__name__}", f"of grblas.{type(vector).__name__}"]
    else:
        name = f"grblas.{type(vector).__name__}"
    # SS, SuiteSparse-specific: format
    keys = ["nvals", "size", "dtype", "format"]
    vals = [vector._nvals, vector._size, vector.dtype.name, get_format(vector)]
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


def vector_header_html(vector, *, mask=None):
    name, keys, vals = vector_info(vector, mask=mask, for_html=True)
    return create_header_html(name, keys, vals)


def _format_html(name, header, df, collapse):
    if has_pandas:
        state = "" if collapse else " open"
        with pd.option_context("display.show_dimensions", False, "display.large_repr", "truncate"):
            details = df._repr_html_()
    else:
        state = ""
        details = "<em>(Install</em> <tt>pandas</tt> <em>to see a preview of the data)</em>"
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
    matrix, *, max_rows=None, min_rows=None, max_columns=None, mask=None, collapse=False
):
    header = matrix_header_html(matrix, mask=mask)
    df = _get_matrix_dataframe(matrix, max_rows, min_rows, max_columns, mask=mask)
    if mask is None:
        name = matrix._name_html
    else:
        name = mask._name_html
    return _format_html(name, header, df, collapse)


def format_vector_html(
    vector, *, max_rows=None, min_rows=None, max_columns=None, mask=None, collapse=False
):
    header = vector_header_html(vector, mask=mask)
    df = _get_vector_dataframe(vector, max_rows, min_rows, max_columns, mask=mask)
    if mask is None:
        name = vector._name_html
    else:
        name = mask._name_html
    return _format_html(name, header, df, collapse)


def format_scalar_html(scalar):
    header = create_header_html("grblas.Scalar", ["value", "dtype"], [scalar.value, scalar.dtype])
    return f'<div class="gb-scalar"><tt>{scalar._name_html}</tt>{header}</div>'


def format_scalar(scalar):
    return create_header(
        "grblas.Scalar",
        ["value", "dtype"],
        [scalar.value, scalar.dtype],
        name=scalar.name,
    )


def get_expr_result(expr, html=False):
    try:
        val = expr._get_value()
    except OutOfMemory:
        arg_string = "Result is too large to compute!"
        if html:
            arg_string = f"<hr><b>{arg_string}</b>"
    else:
        name = val.name
        val.name = "Result"
        if html:
            arg_string = f"<hr>{val._repr_html_()}"
        else:
            arg_string = f"{val}"
        val.name = name
    return arg_string


def _format_expression(expr, header):
    pos_to_arg = {}
    for i, arg in enumerate(expr.args):
        pos = expr.expr_repr.find("{%s" % i)  # -1 if not found
        if pos >= 0:  # pragma: no branch
            pos_to_arg[pos] = arg
    args = [pos_to_arg[pos] for pos in sorted(pos_to_arg)]
    arg_string = "".join(x._repr_html_(collapse=True) for x in args if hasattr(x, "_repr_html_"))
    if config.get("autocompute"):
        arg_string += get_expr_result(expr, html=True)
    return (
        "<div>"
        '<details class="gb-expr-details">'
        '<summary class="gb-expr-summary">'
        f"<b><tt>grblas.{type(expr).__name__}:</tt></b>"
        f"{header}"
        "</summary>"
        '<blockquote class="gb-expr-blockquote">'
        f"{arg_string}"
        "</blockquote>"
        "</details>"
        "<em>"
        "Do <code>expr.new()</code> or <code>other << expr</code> to calculate the expression."
        "</em>"
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
    name = f"grblas.{type(expr).__name__}"
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
    name = f"grblas.{type(expr).__name__}"
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
    name = f"grblas.{type(expr).__name__}"
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
    for key, val in zip(keys, vals):
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


def format_matrix(matrix, *, max_rows=None, min_rows=None, max_columns=None, mask=None):
    name, keys, vals = matrix_info(matrix, mask=mask, for_html=False)
    header = create_header(
        name,
        keys,
        vals,
        lower_border=has_pandas,
        name=matrix.name if mask is None else mask.name,
    )
    if has_pandas:
        df = _get_matrix_dataframe(matrix, max_rows, min_rows, max_columns, mask=mask)
        with pd.option_context("display.show_dimensions", False, "display.large_repr", "truncate"):
            df_repr = df.__repr__()
        return f"{header}\n{df_repr}"
    return header


def format_vector(vector, *, max_rows=None, min_rows=None, max_columns=None, mask=None):
    name, keys, vals = vector_info(vector, mask=mask, for_html=False)
    header = create_header(
        name,
        keys,
        vals,
        lower_border=has_pandas,
        name=vector.name if mask is None else mask.name,
    )
    if has_pandas:
        df = _get_vector_dataframe(vector, max_rows, min_rows, max_columns, mask=mask)
        with pd.option_context("display.show_dimensions", False, "display.large_repr", "truncate"):
            df_repr = df.__repr__()
        return f"{header}\n{df_repr}"
    return header


def _format_infix_expression(expr, header, expr_name):
    arg_string = f"{expr.left._repr_html_(collapse=True)}{expr.right._repr_html_(collapse=True)}"
    if config.get("autocompute") and (
        expr.method_name not in {"ewise_add", "ewise_mult"}
        or expr.left.dtype == BOOL
        and expr.right.dtype == BOOL
    ):
        arg_string += get_expr_result(expr, html=True)
    return (
        "<div>"
        '<details class="gb-expr-details">'
        '<summary class="gb-expr-summary">'
        f"<b><tt>grblas.{type(expr).__name__}:</tt></b>"
        f"{header}"
        "</summary>"
        '<blockquote class="gb-expr-blockquote">'
        f"{arg_string}"
        "</blockquote>"
        "</details>"
        "<em>"
        f"Do <code>op(expr)</code> to create a <tt>{expr.output_type.__name__}</tt> "
        f"for <tt>{expr.method_name}</tt>."
        f"<br>For example: <code>{expr._example_op}({expr_name})</code>"
        "</em>"
        "</div>"
    )


def format_scalar_infix_expression(expr):
    expr_repr = expr._format_expr()
    name = f"grblas.{type(expr).__name__}"
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
    name = f"grblas.{type(expr).__name__}"
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
    name = f"grblas.{type(expr).__name__}"
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
    name = f"grblas.{type(expr).__name__}"
    indices = ", ".join(index._expr_name for index in expr.resolved_indexes.indices)
    expr_repr = f"{expr.parent.name}[{indices}]"
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
    indices = ", ".join(index._expr_name for index in expr.resolved_indexes.indices)
    expr_repr = f"{expr.parent._name_html}[{indices}]"
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
    arg_string = expr.parent._repr_html_(collapse=True)
    if config.get("autocompute"):
        arg_string += get_expr_result(expr, html=True)
    c = expr.output_type.__name__[0]
    c = "M" if c == "M" else c.lower()
    return (
        "<div>"
        '<details class="gb-expr-details">'
        '<summary class="gb-expr-summary">'
        f"<b><tt>grblas.{type(expr).__name__}:</tt></b>"
        f"{header}"
        "</summary>"
        '<blockquote class="gb-expr-blockquote">'
        f"{arg_string}"
        "</blockquote>"
        "</details>"
        "<em>"
        f"This expression may be used to extract or assign a <tt>{expr.output_type.__name__}</tt>."
        f"<br>Example extract: <code>{expr_repr}.new()</code>"
        f"<br>Example assign: <code>{expr_repr} << {'M' if c == 'M' else c.lower()}</code>"
        "</em>"
        "</div>"
    )
