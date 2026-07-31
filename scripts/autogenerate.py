#!/usr/bin/env python
"""Run this script to auto-generate code after modifying automethods or infixmethods.

This can also be done via:

$ python -m graphblas.core.automethods
$ python -m graphblas.core.infixmethods

Auto-methods should be called whenever a new method is added to Scalar, Vector, or Matrix.
It is used to ensure all expressions are able to auto-compute and use the new method.

Modifying infix-methods is much less common, but should be run if you want to modify it.

Pass --check to verify the generated files on disk match a fresh regeneration without
modifying anything. This exits nonzero (and names the drifted files) when they differ,
which is what the drift test uses to catch a hand-edited or stale generated block.

"""

import sys
from pathlib import Path

# For a script, sys.path[0] is the script's own directory rather than the caller's cwd, so
# a bare `import graphblas` resolves to whatever is installed. That coincides with this
# checkout in an ordinary dev setup and diverges in a worktree, where the generator would
# then rewrite, or validate, a tree nobody asked about while looking perfectly normal.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# --check prefixes the package it validated with this, so the drift test can assert it
# checked the checkout it lives in rather than infer correctness from an exit code.
_PACKAGE_LINE = "graphblas package: "

# Files whose auto-generated blocks the generators (re)write. During --check, reads and
# writes are redirected to scratch copies of these so the real tree is never touched.
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
    """Source parsed to an AST dump: identical for two files that differ only in layout."""
    import ast

    return ast.dump(ast.parse(path.read_bytes()))


def check():
    """Regenerate into a scratch tree and report any drift.

    Returns 0 when every generated file matches a fresh regeneration, else 1.

    The comparison is on parsed syntax rather than bytes, which keeps the check
    independent of `black`. The generators shell out to black when it is on PATH and skip
    it when it is not, so a byte comparison reports drift on automethods.py and
    infixmethods.py in every environment lacking it, and black is not a test dependency.
    Layout is already enforced repo-wide by black in pre-commit and the lint job, so the
    drift left for this check to catch is a generated block whose content is stale.
    """
    import shutil
    import tempfile
    from pathlib import Path

    import graphblas
    from graphblas.core import automethods, infixmethods

    # Report the tree actually validated. For a script sys.path[0] is the script's own
    # directory, so without the sys.path fix above this silently checks whichever
    # graphblas is installed; the drift test asserts on this line.
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
    import sys

    if "--check" in sys.argv[1:]:
        sys.exit(check())
    main()
