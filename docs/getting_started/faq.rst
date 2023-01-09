
.. _faq:

Frequently Asked Questions
==========================

FAQs
----

Is there a changelog or release notes?
++++++++++++++++++++++++++++++++++++++

Yes! Notes for each release are currently on GitHub here:

https://github.com/python-graphblas/python-graphblas/releases

How do I cite Python-graphblas?
+++++++++++++++++++++++++++++++

Zenodo provides DOIs for the latest and specific versions of ``python-graphblas`` here:

https://doi.org/10.5281/zenodo.7328791

Follow instructions in "Cite as" in the lower right of the Zenodo webpage.

Where can I find help?
++++++++++++++++++++++

If you think you found a bug, have a feature request, have a question, or have a suggestion
for improving our `documentation <https://python-graphblas.readthedocs.io/en/latest/>`_,
please raise an issue here:

https://github.com/python-graphblas/python-graphblas/issues

More open-ended discussions can be posted here:

https://github.com/python-graphblas/python-graphblas/discussions

You may also join our chat on Discord (in the ``#graphblas`` channel):

https://discord.com/invite/vur45CbwMz

and join our weekly community calls (we're friendly!):

https://github.com/python-graphblas/python-graphblas/issues/247

How and where are decisions made?
+++++++++++++++++++++++++++++++++

We seek consensus and input from all members of the community.
Anyone with an interest in the project can join the community, contribute to the project
design, and participate in the decision making process. We want to hear from you!

We are currently a small team, and major decisions must be agreed upon by both co-creators
`Erik Welch <https://github.com/eriknw>`_ and `Jim Kitchen <https://github.com/jim22k>`_.
PRs must be approved by at least one core developer, and Erik and Jim must both approve
new releases.

We strive for openness, transparency, and institutional neutrality.
Discussions typically occur in our
`weekly community calls <https://github.com/python-graphblas/python-graphblas/issues/247>`_
and documented in `GitHub issues <https://github.com/python-graphblas/python-graphblas/issues>`_.

.. TODO: Notes from our community meetings are here.

As our community grows, we would welcome additional core developers, a Steering Committee,
and other roles to help manage and direct ``python-graphblas``.
We aspire to have (and formalize) governance such as
`NetworkX <https://networkx.org/documentation/stable/developer/nxeps/nxep-0001.html>`_
and `Dask <https://github.com/dask/governance/blob/main/governance.md>`_ have.

How do I report a potential Code of Conduct violation?
++++++++++++++++++++++++++++++++++++++++++++++++++++++

See our code of conduct and instructions for reporting complaints here:

https://github.com/python-graphblas/python-graphblas/blob/main/CODE_OF_CONDUCT.md

What is the relationship between python-graphblas and pygraphblas?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

They are different libraries with similar goals. Both wrap SuiteSparse:GraphBLAS and use the
same underlying ``python-suitesparse-graphblas`` CFFI wrapper, making their underlying objects
compatible with each other.

The biggest difference is in how the C calls are mapped to equivalent Python functions.

For simple expressions, the two libraries are very similar.

.. code-block:: python

    # pygraphblas
    A += 1

.. code-block:: python

    # python-graphblas
    A += 1

For more complex expressions, however, the two libraries diverge significantly.
``pygraphblas`` tends towards the ``numpy`` style of immediate execution and using
keyword arguments to affect the output.

.. code-block:: python

    # pygraphblas
    A.mxm(B.transpose(), mask=A, out=C, accum=FP64.PLUS, semiring=FP64.MIN_PLUS)

``python-graphblas`` uses delayed expressions and keeps the output-affecting arguments
together with the output.

.. code-block:: python

    # python-graphblas
    C(A, accum=binary.plus) << semiring.min_plus(A @ B.T)

``python-graphblas`` also contains additional features, such as the ``Recorder`` and advanced aggregators.

What is the performance penalty of writing algorithms with python-graphblas vs writing them directly in C?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

For large graphs, the performance penalty is negligible as the bulk of the work
of operating on Matrices and Vectors is done purely in C. The overhead is only in the calls made
from Python to C.

For small graphs, the call overhead may become more significant, but smaller graphs usually don't
take much time to compute, so the extra overhead should not be noticeable at a human scale.
