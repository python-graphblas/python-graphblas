import grblas as gb
try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False


def create_header_html(name, keys, vals):
    text = [
        '<div>\n<table style="border:1px solid black">\n'
        '  <tr>\n'
        f'    <td rowspan=2><pre>{name}</pre></td>\n'
    ]
    text.extend(f'    <td><pre>{key}</pre></td>\n' for key in keys)
    text.append('  </tr>\n  <tr>\n')
    text.extend(f'    <td>{val}</td>\n' for val in vals)
    text.append('  </tr>\n</table>\n</div>\n')
    return ''.join(text)


def create_header(name, keys, vals, *, lower_border=False):
    vals = [str(x) for x in vals]
    key_text = []
    val_text = []
    for key, val in zip(keys, vals):
        width = max(len(key), len(val)) + 2
        key_text.append(key.rjust(width))
        val_text.append(val.rjust(width))
    if isinstance(name, str):
        lines = [
            f"{' '*len(name)}{''.join(key_text)}",
            f"{name}{''.join(val_text)}",
        ]
    else:
        name_width = max(map(len, name))
        lines = [f"{' '*name_width}{''.join(key_text)}"]
        lines.extend(line.ljust(name_width) for line in name)
        lines[-1] += ''.join(val_text)
    if lower_border:
        lines.append('-'*len(lines[0]))
    return '\n'.join(lines)


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


def _update_matrix_dataframe(df, matrix, rows, row_offset, columns, column_offset, *, _mask=None):
    if rows is None and columns is None:
        submatrix = matrix
    else:
        if rows is None:
            rows = slice(None)
        if columns is None:
            columns = slice(None)
        if isinstance(matrix, gb.matrix.TransposedMatrix):
            parent = matrix._matrix
            submatrix = gb.Matrix.new(parent.dtype, parent.nrows, parent.ncols)
            submatrix(parent.S)[columns, rows] = 0
            submatrix(submatrix.S) << parent
            if row_offset > 0 or column_offset > 0:
                submatrix = submatrix[column_offset:, row_offset:].new()
        else:
            if _mask is None:
                submatrix = gb.Matrix.new(matrix.dtype, matrix.nrows, matrix.ncols)
                submatrix(matrix.S)[rows, columns] = 0
                submatrix(submatrix.S) << matrix
            else:
                submatrix = gb.Matrix.new('UINT8', matrix.nrows, matrix.ncols)
                if _mask.structure:
                    submatrix(matrix.S)[rows, columns] = 0 if _mask.complement else 1
                else:
                    submatrix(matrix.S)[rows, columns] = 1 if _mask.complement else 0
                    submatrix(matrix.V)[rows, columns] = 0 if _mask.complement else 1
            if row_offset > 0 or column_offset > 0:
                submatrix = submatrix[row_offset:, column_offset:].new()
    if isinstance(matrix, gb.matrix.TransposedMatrix):
        cols, rows, vals = submatrix.to_values()
    else:
        rows, cols, vals = submatrix.to_values()
    df.values[rows, cols] = vals


def _get_matrix_dataframe(matrix, max_rows, min_rows, max_columns, *, _mask=None):
    if max_rows is None:
        max_rows = pd.options.display.max_rows
    if min_rows is None:
        min_rows = pd.options.display.min_rows
    if max_columns is None:
        max_columns = _get_max_columns()
    rows, row_groups = _get_chunk(matrix.nrows, min_rows, max_rows)
    columns, column_groups = _get_chunk(matrix.ncols, max_columns, max_columns)
    df = pd.DataFrame(columns=columns, index=rows)
    for row_group, row_offset in row_groups:
        for column_group, column_offset in column_groups:
            _update_matrix_dataframe(df, matrix, row_group, row_offset, column_group, column_offset, _mask=_mask)
    return df.where(pd.notnull(df), '')


def format_matrix_html(matrix, *, max_rows=None, min_rows=None, max_columns=None, _mask=None):
    if _mask is not None:
        name = f'{type(_mask).__name__}\nof\ngrblas.{type(matrix).__name__}'
    else:
        name = f'grblas.{type(matrix).__name__}'
    keys = ['nvals', 'nrows', 'ncols', 'dtype']
    vals = [matrix.nvals, matrix.nrows, matrix.ncols, matrix.dtype.name]
    header = create_header_html(name, keys, vals)
    if has_pandas:
        df = _get_matrix_dataframe(matrix, max_rows, min_rows, max_columns, _mask=_mask)
        with pd.option_context('display.show_dimensions', False, 'display.large_repr', 'truncate'):
            df_html = df._repr_html_()
        return header + df_html
    return header


def format_matrix(matrix, *, max_rows=None, min_rows=None, max_columns=None, _mask=None):
    if _mask is not None:
        name = [f'{type(_mask).__name__}', f'of grblas.{type(matrix).__name__}']
    else:
        name = f'grblas.{type(matrix).__name__}'
    keys = ['nvals', 'nrows', 'ncols', 'dtype']
    vals = [matrix.nvals, matrix.nrows, matrix.ncols, matrix.dtype.name]
    header = create_header(name, keys, vals, lower_border=has_pandas)
    if has_pandas:
        df = _get_matrix_dataframe(matrix, max_rows, min_rows, max_columns, _mask=_mask)
        with pd.option_context('display.show_dimensions', False, 'display.large_repr', 'truncate'):
            df_repr = df.__repr__()
        return f'{header}\n{df_repr}'
    return header


def _update_vector_dataframe(df, vector, columns, column_offset, *, _mask=None):
    if columns is None:
        subvector = vector
    else:
        if _mask is None:
            subvector = gb.Vector.new(vector.dtype, vector.size)
            subvector(vector.S)[columns] = 0
            subvector(subvector.S) << vector
        else:
            subvector = gb.Vector.new('UINT8', vector.size)
            if _mask.structure:
                subvector(vector.S)[columns] = 0 if _mask.complement else 1
            else:
                subvector(vector.S)[columns] = 1 if _mask.complement else 0
                subvector(vector.V)[columns] = 0 if _mask.complement else 1
        if column_offset > 0:
            subvector = subvector[column_offset:].new()
    cols, vals = subvector.to_values()
    df.values[0, cols] = vals


def _get_vector_dataframe(vector, max_columns, *, _mask=None):
    if max_columns is None:
        max_columns = _get_max_columns()
    columns, column_groups = _get_chunk(vector.size, max_columns, max_columns)
    df = pd.DataFrame(columns=columns, index=[''])
    for column_group, column_offset in column_groups:
        _update_vector_dataframe(df, vector, column_group, column_offset, _mask=_mask)
    return df.where(pd.notnull(df), '')


def format_vector_html(vector, *, max_columns=None, _mask=None):
    if _mask is not None:
        name = f'{type(_mask).__name__}\nof\ngrblas.{type(vector).__name__}'
    else:
        name = f'grblas.{type(vector).__name__}'
    keys = ['nvals', 'size', 'dtype']
    vals = [vector.nvals, vector.size, vector.dtype.name]
    header = create_header_html(name, keys, vals)
    if has_pandas:
        df = _get_vector_dataframe(vector, max_columns, _mask=_mask)
        with pd.option_context('display.show_dimensions', False, 'display.large_repr', 'truncate'):
            df_html = df._repr_html_()
        return header + df_html
    return header


def format_vector(vector, *, max_columns=None, _mask=None):
    if _mask is not None:
        name = [f'{type(_mask).__name__}', 'of grblas.{type(vector).__name__}']
    else:
        name = f'grblas.{type(vector).__name__}'
    keys = ['nvals', 'size', 'dtype']
    vals = [vector.nvals, vector.size, vector.dtype.name]
    header = create_header(name, keys, vals, lower_border=has_pandas)
    if has_pandas:
        df = _get_vector_dataframe(vector, max_columns, _mask=_mask)
        with pd.option_context('display.show_dimensions', False, 'display.large_repr', 'truncate'):
            df_repr = df.__repr__()
        return f'{header}\n{df_repr}'
    return header
