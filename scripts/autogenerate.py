#!/usr/bin/env python
"""This script is used to auto-generate code.

This can also be done via:

$ python -m graphblas._automethods
$ python -m graphblas._infixmethods

Auto-methods should be called whenever a new method is added to Scalar, Vector, or Matrix.
It is used to ensure all expressions are able to auto-compute and use the new method.

Modifying infix-methods is much less common, but should be run if you want to modify it.

"""


def main():
    from graphblas._automethods import _main as auto_main
    from graphblas._infixmethods import _main as infix_main

    auto_main()
    infix_main()


if __name__ == "__main__":
    main()
