def get_order(order):
    val = order.lower()
    if val in {"c", "row", "rows", "rowwise"}:
        return "rowwise"
    elif val in {"f", "col", "cols", "column", "columns", "columnwise"}:
        return "columnwise"
    else:
        raise ValueError(
            f"Bad value for order: {order!r}.  "
            'Expected "rowwise", "columnwise", "rows", "columns", "C", or "F"'
        )
