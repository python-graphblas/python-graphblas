"""python-graphblas benchmark suite (airspeed velocity).

Benchmark modules live alongside this file. Shared data builders are in
``common.py``. Each benchmark module imports ``common`` with a dual path so it
works whether asv imports the files as a package or adds ``benchmark_dir`` to
``sys.path`` and imports them flat:

    try:
        from . import common
    except ImportError:  # imported flat, not as a package
        import common
"""
