#!/usr/bin/env python
"""Run this script to auto-generate code after modifying automethods or infixmethods.

This can also be done via:

$ python -m graphblas.core.automethods
$ python -m graphblas.core.infixmethods

Auto-methods should be called whenever a new method is added to Scalar, Vector, or Matrix.
It is used to ensure all expressions are able to auto-compute and use the new method.

Modifying infix-methods is much less common, but should be run if you want to modify it.

With --check, it regenerates the files in a scratch directory and compares them with the
files in graphblas/core, which it leaves alone. It names the files that differ and exits
with 1 if any do. graphblas/tests/test_autogenerate_check.py runs it.

"""

import sys
from pathlib import Path

# A script's sys.path[0] is its own directory, so ``import graphblas`` would find the
# installed package, which in a git worktree is another checkout. Use this checkout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# --check prints the package it checked after this, so that the test can assert that it
# checked its own checkout
_PACKAGE_LINE = "graphblas package: "

# The files with blocks the generators write; --check copies them to a scratch directory
_GENERATED_FILES = (
    "automethods.py",
    "infixmethods.py",
    "scalar.py",
    "vector.py",
    "matrix.py",
    "infix.py",
)


def main():
    from graphblas.core.automethods import _main as auto_main
    from graphblas.core.infixmethods import _main as infix_main

    auto_main()
    infix_main()


def _parsed(path):
    """Return the AST dump of a file: the same for two files that differ only in layout."""
    import ast

    return ast.dump(ast.parse(path.read_bytes()))


def check():
    """Regenerate the files in a scratch directory, and return 1 if any differs, else 0.

    The files are compared as parsed syntax, not bytes, so the check does not need black:
    the generators run black only when it is installed, and black is not a test
    dependency. pre-commit checks the layout.
    """
    import shutil
    import tempfile

    import graphblas
    from graphblas.core import automethods, infixmethods

    print(f"{_PACKAGE_LINE}{Path(graphblas.__file__).resolve().parent}")

    src_dir = Path(automethods.__file__).parent
    with tempfile.TemporaryDirectory(prefix="autogen_check_") as tmp:
        scratch = Path(tmp)
        for name in _GENERATED_FILES:
            shutil.copyfile(src_dir / name, scratch / name)
        automethods._main(base_dir=scratch, callblack=False)
        infixmethods._main(base_dir=scratch, callblack=False)

        drifted = [
            name for name in _GENERATED_FILES if _parsed(scratch / name) != _parsed(src_dir / name)
        ]

    if drifted:
        print("Auto-generated code is out of date; run `python scripts/autogenerate.py`.")
        print("Drifted file(s):")
        for name in drifted:
            print(f"  graphblas/core/{name}")
        return 1
    print("Auto-generated code is up to date.")
    return 0


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        sys.exit(check())
    main()
