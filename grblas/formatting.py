from .matrix import Matrix, TransposedMatrix
from .vector import Vector

try:
    import pandas as pd

    has_pandas = True
except ImportError:  # pragma: no cover
    has_pandas = False


def _update_matrix_dataframe(df, matrix, rows, row_offset, columns, column_offset, *, mask=None):
    if rows is None and columns is None:
        if mask is None:
            submatrix = matrix
        else:
            submatrix = Matrix.new("UINT8", matrix.nrows, matrix.ncols, name="")
            if mask.structure:
                submatrix(matrix.S)[:, :] = 0 if mask.complement else 1
            else:
                submatrix(matrix.S)[:, :] = 1 if mask.complement else 0
                submatrix(matrix.V)[:, :] = 0 if mask.complement else 1
    else:
        if rows is None:
            rows = slice(None)
        if columns is None:
            columns = slice(None)
        if type(matrix) is TransposedMatrix:
            parent = matrix._matrix
            submatrix = Matrix.new(parent.dtype, parent.nrows, parent.ncols, name="")
            submatrix(parent.S)[columns, rows] = 0
            submatrix(submatrix.S) << parent
            if row_offset > 0 or column_offset > 0:
                submatrix = submatrix[column_offset:, row_offset:].new(name="")
            submatrix = submatrix.T
        else:
            if mask is None:
                submatrix = Matrix.new(matrix.dtype, matrix.nrows, matrix.ncols, name="")
                submatrix(matrix.S)[rows, columns] = 0
                submatrix(submatrix.S) << matrix
            else:
                submatrix = Matrix.new("UINT8", matrix.nrows, matrix.ncols, name="")
                if mask.structure:
                    submatrix(matrix.S)[rows, columns] = 0 if mask.complement else 1
                else:
                    submatrix(matrix.S)[rows, columns] = 1 if mask.complement else 0
                    submatrix(matrix.V)[rows, columns] = 0 if mask.complement else 1
            if row_offset > 0 or column_offset > 0:
                submatrix = submatrix[row_offset:, column_offset:].new(name="")
    rows, cols, vals = submatrix.to_values()
    df.values[rows, cols] = vals


def _update_vector_dataframe(df, vector, columns, column_offset, *, mask=None):
    if columns is None:
        if mask is None:
            subvector = vector
        else:
            subvector = Vector.new("UINT8", vector.size, name="")
            if mask.structure:
                subvector(vector.S)[:] = 0 if mask.complement else 1
            else:
                subvector(vector.S)[:] = 1 if mask.complement else 0
                subvector(vector.V)[:] = 0 if mask.complement else 1
    else:
        if mask is None:
            subvector = Vector.new(vector.dtype, vector.size, name="")
            subvector(vector.S)[columns] = 0
            subvector(subvector.S) << vector
        else:
            subvector = Vector.new("UINT8", vector.size, name="")
            if mask.structure:
                subvector(vector.S)[columns] = 0 if mask.complement else 1
            else:
                subvector(vector.S)[columns] = 1 if mask.complement else 0
                subvector(vector.V)[columns] = 0 if mask.complement else 1
        if column_offset > 0:
            subvector = subvector[column_offset:].new(name="")
    cols, vals = subvector.to_values()
    df.values[0, cols] = vals


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
        chunk_groups = [(None, 0)]
    else:
        half = min_length // 2
        first_chunk = list(range(half))
        second_chunk = list(range(length - half, length))
        chunk = list(range(max_length + 1))
        chunk[-half:] = second_chunk
        chunk_groups = [(first_chunk, 0), (second_chunk, length - len(chunk))]
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
    rows, row_groups = _get_chunk(matrix.nrows, min_rows, max_rows)
    columns, column_groups = _get_chunk(matrix.ncols, max_columns, max_columns)
    df = pd.DataFrame(columns=columns, index=rows)
    for row_group, row_offset in row_groups:
        for column_group, column_offset in column_groups:
            _update_matrix_dataframe(
                df, matrix, row_group, row_offset, column_group, column_offset, mask=mask,
            )
    return df.where(pd.notnull(df), "")


def _get_vector_dataframe(vector, max_columns, *, mask=None):
    if not has_pandas:
        return
    if max_columns is None:  # pragma: no branch
        max_columns = _get_max_columns()
    columns, column_groups = _get_chunk(vector.size, max_columns, max_columns)
    df = pd.DataFrame(columns=columns, index=[""])
    for column_group, column_offset in column_groups:
        _update_vector_dataframe(df, vector, column_group, column_offset, mask=mask)
    return df.where(pd.notnull(df), "")


def matrix_info(matrix, *, mask=None, for_html=True):
    if mask is not None:
        if for_html:
            name = f"{type(mask).__name__}\nof\ngrblas.{type(matrix).__name__}"
        else:
            name = [f"{type(mask).__name__}", f"of grblas.{type(matrix).__name__}"]
    else:
        name = f"grblas.{type(matrix).__name__}"
    keys = ["nvals", "nrows", "ncols", "dtype"]
    vals = [matrix.nvals, matrix.nrows, matrix.ncols, matrix.dtype.name]
    return name, keys, vals


def vector_info(vector, *, mask=None, for_html=True):
    if mask is not None:
        if for_html:
            name = f"{type(mask).__name__}\nof\ngrblas.{type(vector).__name__}"
        else:
            name = [f"{type(mask).__name__}", f"of grblas.{type(vector).__name__}"]
    else:
        name = f"grblas.{type(vector).__name__}"
    keys = ["nvals", "size", "dtype"]
    vals = [vector.nvals, vector.size, vector.dtype.name]
    return name, keys, vals


def create_header_html(name, keys, vals):
    text = [
        '<div>\n<table style="border:1px solid black; max-width:100%;">\n'
        "  <tr>\n"
        f'    <td rowspan="2" style="line-height:100%"><pre>{name}</pre></td>\n'
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


def _format_html(name, header, df):
    if has_pandas:
        state = " open"
        with pd.option_context("display.show_dimensions", False, "display.large_repr", "truncate"):
            details = df._repr_html_()
    else:
        state = ""
        details = "<em>(Install</em> <tt>pandas</tt> <em>to see a preview of the data)</em>"
    return (
        "<div>"
        f"<details{state}>"
        '<summary style="display:list-item; outline:none;">'
        f"<tt>{name}</tt>{header}"
        "</summary>"
        f"{details}"
        "</details>"
        "</div>"
    )


def format_matrix_html(matrix, *, max_rows=None, min_rows=None, max_columns=None, mask=None):
    header = matrix_header_html(matrix, mask=mask)
    df = _get_matrix_dataframe(matrix, max_rows, min_rows, max_columns, mask=mask)
    if mask is None:
        name = matrix._name_html
    else:
        name = mask._name_html
    return _format_html(name, header, df)


def format_vector_html(vector, *, max_columns=None, mask=None):
    header = vector_header_html(vector, mask=mask)
    df = _get_vector_dataframe(vector, max_columns, mask=mask)
    if mask is None:
        name = vector._name_html
    else:
        name = mask._name_html
    return _format_html(name, header, df)


def format_scalar_html(scalar):
    header = create_header_html("grblas.Scalar", ["value", "dtype"], [scalar.value, scalar.dtype])
    return f"<div><tt>{scalar._name_html}</tt>{header}</div>"


def format_scalar(scalar):
    return create_header(
        "grblas.Scalar", ["value", "dtype"], [scalar.value, scalar.dtype], name=scalar.name,
    )


def _format_expression(expr, header):
    pos_to_arg = {}
    for i, arg in enumerate(expr.args):
        pos = expr.expr_repr.find("{%s" % i)  # -1 if not found
        if pos >= 0:  # pragma: no branch
            pos_to_arg[pos] = arg
    args = [pos_to_arg[pos] for pos in sorted(pos_to_arg)]
    arg_string = "".join(x._repr_html_() for x in args if hasattr(x, "_repr_html_"))
    return (
        '<div style="padding:4px;">'
        "<details>"
        '<summary style="display:list-item; outline:none;">'
        f"<b><tt>grblas.{type(expr).__name__}:</tt></b>"
        f"{header}"
        "</summary>"
        "<blockquote>"
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
        expr_html, ["nrows", "ncols", "dtype"], [expr.nrows, expr.ncols, expr.dtype]
    )
    return _format_expression(expr, header)


def format_matrix_expression(expr):
    expr_repr = expr._format_expr()
    name = f"grblas.{type(expr).__name__}"
    header = create_header(
        expr_repr,
        ["nrows", "ncols", "dtype"],
        [expr.nrows, expr.ncols, expr.dtype],
        name=name,
        quote=False,
    )
    return f"{header}\n\nDo expr.new() or other << expr to calculate the expression."


def format_vector_expression_html(expr):
    expr_html = expr._format_expr_html()
    header = create_header_html(expr_html, ["size", "dtype"], [expr.size, expr.dtype])
    return _format_expression(expr, header)


def format_vector_expression(expr):
    expr_repr = expr._format_expr()
    name = f"grblas.{type(expr).__name__}"
    header = create_header(
        expr_repr, ["size", "dtype"], [expr.size, expr.dtype], name=name, quote=False
    )
    return f"{header}\n\nDo expr.new() or other << expr to calculate the expression."


def format_scalar_expression_html(expr):
    expr_html = expr._format_expr_html()
    header = create_header_html(expr_html, ["dtype"], [expr.dtype])
    return _format_expression(expr, header)


def format_scalar_expression(expr):
    expr_repr = expr._format_expr()
    name = f"grblas.{type(expr).__name__}"
    header = create_header(expr_repr, ["dtype"], [expr.dtype], name=name, quote=False)
    return f"{header}\n\nDo expr.new() or other << expr to calculate the expression."


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
        name, keys, vals, lower_border=has_pandas, name=matrix.name if mask is None else mask.name,
    )
    if has_pandas:
        df = _get_matrix_dataframe(matrix, max_rows, min_rows, max_columns, mask=mask)
        with pd.option_context("display.show_dimensions", False, "display.large_repr", "truncate"):
            df_repr = df.__repr__()
        return f"{header}\n{df_repr}"
    return header


def format_vector(vector, *, max_columns=None, mask=None):
    name, keys, vals = vector_info(vector, mask=mask, for_html=False)
    header = create_header(
        name, keys, vals, lower_border=has_pandas, name=vector.name if mask is None else mask.name,
    )
    if has_pandas:
        df = _get_vector_dataframe(vector, max_columns, mask=mask)
        with pd.option_context("display.show_dimensions", False, "display.large_repr", "truncate"):
            df_repr = df.__repr__()
        return f"{header}\n{df_repr}"
    return header
