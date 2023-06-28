import numpy as np

from ..core.matrix import Matrix
from ..core.utils import output_type
from ..core.vector import Vector

_AwkwardDoublyCompressedMatrix = None


def from_awkward(A, *, name=None):
    """Create a Matrix or Vector from an Awkward Array.

    The Awkward Array must have top-level parameters: format, shape

    The Awkward Array must have top-level attributes based on format:
    - vec/csr/csc: values, indices
    - hypercsr/hypercsc: values, indices, offset_labels

    Parameters
    ----------
    A : awkward.Array
        Awkward Array with values and indices
    name : str, optional
        Name of resulting Matrix or Vector

    Returns
    -------
    Vector or Matrix
    """
    params = A.layout.parameters
    if missing := {"format", "shape"} - params.keys():
        raise ValueError(f"Missing parameters: {missing}")
    format = params["format"]
    shape = params["shape"]

    if len(shape) == 1:
        if format != "vec":
            raise ValueError(f"Invalid format for Vector: {format}")
        return Vector.from_coo(
            A.indices.layout.data, A.values.layout.data, size=shape[0], name=name
        )
    nrows, ncols = shape
    values = A.values.layout.content.data
    indptr = A.values.layout.offsets.data
    if format == "csr":
        cols = A.indices.layout.content.data
        return Matrix.from_csr(indptr, cols, values, ncols=ncols, name=name)
    if format == "csc":
        rows = A.indices.layout.content.data
        return Matrix.from_csc(indptr, rows, values, nrows=nrows, name=name)
    if format == "hypercsr":
        rows = A.offset_labels.layout.data
        cols = A.indices.layout.content.data
        return Matrix.from_dcsr(rows, indptr, cols, values, nrows=nrows, ncols=ncols, name=name)
    if format == "hypercsc":
        cols = A.offset_labels.layout.data
        rows = A.indices.layout.content.data
        return Matrix.from_dcsc(cols, indptr, rows, values, nrows=nrows, ncols=ncols, name=name)
    raise ValueError(f"Invalid format for Matrix: {format}")


def to_awkward(A, format=None):
    """Create an Awkward Array from a GraphBLAS Matrix.

    Parameters
    ----------
    A : Matrix or Vector
        GraphBLAS object to be converted
    format : str {'csr', 'csc', 'hypercsr', 'hypercsc', 'vec}
        Default format is csr for Matrix; vec for Vector

    The Awkward Array will have top-level attributes based on format:
    - vec/csr/csc: values, indices
    - hypercsr/hypercsc: values, indices, offset_labels

    Top-level parameters will also be set: format, shape

    Returns
    -------
    awkward.Array

    """
    try:
        # awkward version 1
        # MAINT: we can probably drop awkward v1 at the end of 2024 or 2025
        import awkward._v2 as ak
        from awkward._v2.forms.listoffsetform import ListOffsetForm
        from awkward._v2.forms.numpyform import NumpyForm
        from awkward._v2.forms.recordform import RecordForm
    except ImportError:
        # awkward version 2
        import awkward as ak
        from awkward.forms.listoffsetform import ListOffsetForm
        from awkward.forms.numpyform import NumpyForm
        from awkward.forms.recordform import RecordForm

    out_type = output_type(A)
    if format is None:
        format = "vec" if out_type is Vector else "csr"
    format = format.lower()
    classname = None

    if out_type is Vector:
        if format != "vec":
            raise ValueError(f"Invalid format for Vector: {format}")
        size = A.nvals
        indices, values = A.to_coo()
        form = RecordForm(
            contents=[
                NumpyForm(A.dtype.np_type.name, form_key="node1"),
                NumpyForm("int64", form_key="node0"),
            ],
            fields=["values", "indices"],
        )
        d = {"node0-data": indices, "node1-data": values}

    elif out_type is Matrix:
        if format == "csr":
            indptr, cols, values = A.to_csr()
            d = {"node3-data": cols}
            size = A.nrows
        elif format == "csc":
            indptr, rows, values = A.to_csc()
            d = {"node3-data": rows}
            size = A.ncols
        elif format == "hypercsr":
            rows, indptr, cols, values = A.to_dcsr()
            d = {"node3-data": cols, "node5-data": rows}
            size = len(rows)
        elif format == "hypercsc":
            cols, indptr, rows, values = A.to_dcsc()
            d = {"node3-data": rows, "node5-data": cols}
            size = len(cols)
        else:
            raise ValueError(f"Invalid format for Matrix: {format}")
        d["node1-offsets"] = indptr
        d["node4-data"] = np.ascontiguousarray(values)

        form = ListOffsetForm(
            "i64",
            RecordForm(
                contents=[
                    NumpyForm("int64", form_key="node3"),
                    NumpyForm(A.dtype.np_type.name, form_key="node4"),
                ],
                fields=["indices", "values"],
            ),
            form_key="node1",
        )
        if format.startswith("hyper"):
            global _AwkwardDoublyCompressedMatrix
            if _AwkwardDoublyCompressedMatrix is None:  # pylint: disable=used-before-assignment
                # Define behaviors to make all fields function at the top-level
                @ak.behaviors.mixins.mixin_class(ak.behavior)
                class _AwkwardDoublyCompressedMatrix:
                    @property
                    def values(self):  # pragma: no branch (???)
                        return self.data.values

                    @property
                    def indices(self):  # pragma: no branch (???)
                        return self.data.indices

            form = RecordForm(
                contents=[
                    form,
                    NumpyForm("int64", form_key="node5"),
                ],
                fields=["data", "offset_labels"],
            )
            classname = "_AwkwardDoublyCompressedMatrix"

    else:
        raise TypeError(f"A must be a Matrix or Vector, found {type(A)}")

    ret = ak.from_buffers(form, size, d)
    ret = ak.with_parameter(ret, "format", format)
    ret = ak.with_parameter(ret, "shape", list(A.shape))
    if classname:
        ret = ak.with_name(ret, classname)
    return ret
