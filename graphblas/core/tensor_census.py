"""Counters of the Tensor's decisions, for a census of its engines, strategies and heuristics.

Private, and off by default: each decision point tests ``tensor_census.on`` before doing
anything else, so off it costs one attribute read. On (``start()``, or the environment variable
``GRAPHBLAS_TENSOR_CENSUS`` naming a file that the records are appended to as JSON lines at
exit), each decision appends a record: the decision point, the choice, the estimated cost of
each option where the point chooses by cost, and details. ``force[point] = label`` makes a
point that chooses by cost take that option wherever it is offered, to time the alternatives.
"""

import atexit
import json
import os
from pathlib import Path

on = False
records = []
force = {}
context = {}  # merged into every record (a census driver sets the case being run)


def start():
    global on
    records.clear()
    on = True


def stop():
    """Turn the census off; return its records."""
    global on
    on = False
    return list(records)


def note(point, choice, costs=None, **info):
    records.append({"point": point, "choice": choice, "costs": costs, **context, **info})


def choose(point, costs, labels, best, **info):
    """Record a choice among options of estimated ``costs`` (ns) named ``labels``, of which
    ``best`` (an index) is the cheapest; return the index to take: ``best``, or the option
    ``force[point]`` names when it is offered.
    """
    named = {}
    for label, cost in zip(labels, costs, strict=True):
        if label not in named or cost < named[label]:
            named[label] = cost
    i = best
    if (want := force.get(point)) is not None and want in labels:
        i = labels.index(want)
    note(point, labels[i], named, forced=i != best, **info)
    return i


def _dump(path):
    with Path(path).open("a") as f:
        f.writelines(json.dumps(rec, default=str) + "\n" for rec in records)


if path := os.environ.get("GRAPHBLAS_TENSOR_CENSUS"):
    start()
    atexit.register(_dump, path)
