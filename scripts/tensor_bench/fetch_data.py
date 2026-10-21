"""Download the benchmark and demo data from the original sources, and check it.

    python fetch_data.py --list                    # what there is, sizes, licenses
    python fetch_data.py frostt-uber hetionet      # fetch some
    python fetch_data.py all                       # everything (about 600 MB)
    python fetch_data.py frostt-nell2 --copy-from /path/to/a/directory/with/the/files

Files go to the data directory (``--data-dir``, else ``TENSOR_BENCH_DATA``, else ``data/``
next to this script), one subdirectory per source. Every file is checked against the SHA-256
recorded below, and a file that does not match is deleted. Nothing is redistributed: each
file comes from its publisher, and each dataset's license and citation are printed when it
is fetched and written to ``SOURCES-<name>.md`` in its subdirectory. Cite the sources you use.

FROSTT tensors are also converted once to ``frostt/cache/<name>.npz`` for the benchmark
harness (``harness/datasets.py`` does that on first use if you skip it here).
"""

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import paths

sys.path.insert(0, str(Path(__file__).resolve().parent))

FROSTT = "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/"
FROSTT_CITE = (
    "Shaden Smith, Jee W. Choi, Jiajia Li, Richard Vuduc, Jongsoo Park, Xing Liu and George "
    "Karypis, "
    "FROSTT: The Formidable Repository of Open Sparse Tensors and Tools, 2017, http://frostt.io/ ; "
    "and the tensor's original source, as its FROSTT page gives it"
)
FROSTT_LICENSE = "as stated on the tensor's FROSTT page and by its original source"
LAGRAPH = (
    "https://raw.githubusercontent.com/GraphBLAS/LAGraph/7543de1f310371c540c23ff5da82705aa4e94ea0/"
)


def _frostt(name, path, size, sha, page, what):
    return {
        "dir": "frostt",
        "files": [(f"{name}.tns.gz", FROSTT + path, size, sha)],
        "page": page,
        "what": what,
        "license": FROSTT_LICENSE,
        "cite": FROSTT_CITE,
        "frostt": name,
    }


# name: dir, files (local name, url, bytes, sha256), page, what, license, citation
DATASETS = {
    "frostt-uber": _frostt(
        "uber",
        "uber-pickups/uber.tns.gz",
        10_073_910,
        "ba70964e417823ea9ac63bb12219bf74c456b52a0c6fa8c1b479289b621f08e8",
        "http://frostt.io/tensors/uber-pickups/",
        "4-d, 183 x 24 x 1140 x 1717, 3.3M entries (date x hour x latitude x longitude, pickup "
        "counts)",
    ),
    "frostt-nips": _frostt(
        "nips",
        "nips/nips.tns.gz",
        9_354_154,
        "57f46bf18d5b2d3cf8f718418aa9a8dbbfd27648d39e9850edf3ec45de0a53be",
        "http://frostt.io/tensors/nips/",
        "4-d, 2482 x 2862 x 14036 x 17, 3.1M entries (paper x author x word x year, counts)",
    ),
    "frostt-chicago": _frostt(
        "chicago-crime-comm",
        "chicago-crime/comm/chicago-crime-comm.tns.gz",
        15_584_935,
        "e05a43d34da729efb972254c41421e5d07e59b9278ef25e25e4b5c743c3eb7d8",
        "http://frostt.io/tensors/chicago-crime/",
        "4-d, 6186 x 24 x 77 x 32, 5.3M entries (date x hour x community area x crime type, "
        "counts)",
    ),
    "frostt-nell2": _frostt(
        "nell-2",
        "nell/nell-2.tns.gz",
        275_306_707,
        "4a6b8599e7a52765c87e4cf6defbda1456ea5f43c635150d834b1a3722ab15fd",
        "http://frostt.io/tensors/nell-2/",
        "3-d, 12092 x 9184 x 28818, 76.9M entries (entity x relation x entity, counts)",
    ),
    "frostt-enron": _frostt(
        "enron",
        "enron/enron.tns.gz",
        178_933_828,
        "6e0da0fff11ae69d0354580d7c1b86efc5bfa6854c420e34da4a0065fb2b34da",
        "http://frostt.io/tensors/enron/",
        "4-d, 6066 x 5699 x 244268 x 1176, 54.2M entries (sender x receiver x word x date, counts)",
    ),
    "frostt-lbnl": _frostt(
        "lbnl-network",
        "lbnl-network/lbnl-network.tns.gz",
        6_943_849,
        "803931041984d484e2aa3a87399196d53972f336b40c706105017648c4bad741",
        "http://frostt.io/tensors/lbnl-network/",
        "5-d, 1605 x 4198 x 1631 x 4209 x 868131, 1.7M entries (sender IP x port x receiver IP x "
        "port x "
        "time); the traffic-windows demo",
    ),
    "hetionet": {
        "dir": "hetionet",
        "files": [
            (
                "hetionet-v1.0-edges.sif.gz",
                "https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz",
                12_401_407,
                "611f0411ac666be4e0270e54bbca67407e14720bfc3099311f0f5eec25c5a947",
            ),
            (
                "hetionet-v1.0-nodes.tsv",
                "https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-nodes.tsv",
                2_509_984,
                "3ad9eb86c031df8bc7d5efefa6edc93562182eb9a6a71e5129d3f18b9d42c50e",
            ),
        ],
        "page": "https://het.io/ (https://github.com/hetio/hetionet)",
        "what": "Hetionet v1.0: 47,031 nodes of 11 kinds, 2,250,197 edges of 24 metaedges; the "
        "metapath and RPQ demos",
        "license": "CC0 for the original content; the integrated resources keep their own "
        "licenses, recorded "
        "per node and edge upstream",
        "cite": "Himmelstein DS, et al., Systematic integration of biomedical knowledge "
        "prioritizes drugs for "
        "repurposing, eLife 2017;6:e26726, https://doi.org/10.7554/eLife.26726",
    },
    "eu-airlines": {
        "dir": "eu_airlines",
        "files": [
            (
                "eu_airlines.csv.zip",
                "https://networks.skewed.de/net/eu_airlines/files/eu_airlines.csv.zip",
                24_291,
                "c327d9c434ef2d4d09a798699adb6fab4a609aa27323d76f62db2b8ccdd599c4",
            ),
        ],
        "unzip": {
            "edges.csv": "23ba41a3d5aba70f8714de28131462e268b1ca930447a0aaff4c3f1cde8c2a08",
            "nodes.csv": "7f088e12846d97a6d35682c4d5eb3d59fddb295247285bf2fda0e16c63f0d730",
            "gprops.csv": "74407bf1f6cf0f6583225fb33ab6b23134a395bf5080a3621f926ec5b74d16f4",
        },
        "page": "https://networks.skewed.de/net/eu_airlines (Netzschleuder)",
        "what": "EU airline multiplex: 450 airports, 37 airline layers, 3,588 undirected edges; "
        "the multiplex "
        "PageRank demo",
        "license": "none stated by Netzschleuder or the original page; research data, cite it",
        "cite": "A. Cardillo, J. Gomez-Gardenes, M. Zanin, M. Romance, D. Papo, F. del Pozo and "
        "S. Boccaletti, "
        "Emergence of network features from multiplexity, Scientific Reports 3, 1344 (2013), "
        "https://doi.org/10.1038/srep01344",
    },
    "higgs": {
        "dir": "higgs_twitter",
        "files": [
            (
                f"higgs-{n}_network.edgelist.gz",
                f"https://snap.stanford.edu/data/higgs-{n}_network.edgelist.gz",
                size,
                sha,
            )
            for n, size, sha in (
                (
                    "social",
                    54_597_265,
                    "8e5ea7642a018d9aab2dba0c6a59832c278cecf3895661ec8ff0d65360d40265",
                ),
                (
                    "retweet",
                    1_957_437,
                    "4097edebd539043cfc62d9b4ba67676f79523f6dbe896d345fcc127dec3e936e",
                ),
                (
                    "reply",
                    203_054,
                    "449f13a5d0eebd1dff9874db4db662e8a83f9b789e0c68faec871cf069c8dea6",
                ),
                (
                    "mention",
                    885_911,
                    "df115f39619ec9736c460faa176810e12813a8c344e239ec53a439073f05173f",
                ),
            )
        ],
        "page": "https://snap.stanford.edu/data/higgs-twitter.html (SNAP)",
        "what": "Higgs Twitter, 4 directed layers over 456,626 users (15.4M edges); the multiplex "
        "PageRank demo",
        "license": "none stated by SNAP; published for research use, cite it",
        "cite": "M. De Domenico, A. Lima, P. Mougel and M. Musolesi, The Anatomy of a Scientific "
        "Rumor, "
        "Scientific Reports 3, 2980 (2013)",
    },
    "lagraph-rpq": {
        "dir": "lagraph_rpq",
        "files": [
            (f, LAGRAPH + "data/rpq_data/" + f, None, sha)
            for f, sha in (
                ("a.mtx", "f1d9055785d9060b22d795a7d0136f81dbdbd9b8ef70749d62c38ec9ed13657b"),
                ("b.mtx", "0a3aed6495c2cfe178bbb700f27cca9cd2dec3c1aa3aedf3a7daab0079444a94"),
                ("1_a.mtx", "59be779a31920280f2d7a9a3d275dd7cccf1c0f8d5824d187d55c338956be11a"),
                ("1_meta.txt", "45e81aedb0e59022bf47117fc45f980574552a0600aa3a09e36daddcf35b0342"),
                (
                    "1_sources.txt",
                    "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",
                ),
                ("2_a.mtx", "d58dfecd88123ac7761220af5db26f77515aa16d767ef4a0dc15ab1b849111b6"),
                ("2_b.mtx", "1651475ca444d7f123415e025ee554a7dc33d420d00b078a6b59c43cd2167b90"),
                ("2_meta.txt", "f5c5e583808c61ef64c7c606dbd9f107efadadc78ef231dae68662e333a2d573"),
                (
                    "2_sources.txt",
                    "53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3",
                ),
                ("3_a.mtx", "6c0161e737712071ddf9f3a67a5eab2f4804e5619f7cb924df29eefaf92b6773"),
                ("3_b.mtx", "6c0161e737712071ddf9f3a67a5eab2f4804e5619f7cb924df29eefaf92b6773"),
                ("3_meta.txt", "f5c5e583808c61ef64c7c606dbd9f107efadadc78ef231dae68662e333a2d573"),
                (
                    "3_sources.txt",
                    "1f2bbb6596fac0e3a9683e8412c862103a7c31244cae330a31ca602bc922e69a",
                ),
                ("4_b.mtx", "aace9fef0f6567188718b0c2a46ae47fe022640b13d84cf2d743f637558a6fc7"),
                ("4_meta.txt", "0711288ace64511aae3eaa6d79a2f4d73af9e978cf7a40c3ff688a134297c6d1"),
                (
                    "4_sources.txt",
                    "0b6693b0fa788f42f92edb0721ac4da7e1b22ea0c796913a2c5e83b6271cfaf5",
                ),
            )
        ]
        + [
            (
                "test_RegularPathQuery.c",
                LAGRAPH + "experimental/test/test_RegularPathQuery.c",
                None,
                "886f2c95979460023137bae3ab334ddaa76ae5a796ef716fed96d7fb209e9e94",
            ),
            (
                "LICENSE.LAGraph",
                LAGRAPH + "LICENSE",
                None,
                "d92d9b022d4b8ee7baa11e750f396ab333f257b08d389d5615997e13eec70c79",
            ),
        ],
        "page": "https://github.com/GraphBLAS/LAGraph (commit 7543de1f, data/rpq_data and its "
        "test)",
        "what": "LAGraph's regular path query test cases: an 8-node graph with labels a and b, "
        "four NFAs, "
        "their sources, and the test with the expected outputs; the RPQ demo",
        "license": "BSD-2-Clause (LAGraph)",
        "cite": "LAGraph, https://github.com/GraphBLAS/LAGraph ; the RPQ algorithm: G. Belyanin "
        "and S. Grigoriev, "
        "arXiv:2412.10287",
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dst):
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(
        url, headers={"User-Agent": "python-graphblas tensor_bench fetch_data"}
    )
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dst)


def fetch(name, copy_from=None):
    spec = DATASETS[name]
    d = paths.data_dir() / spec["dir"]
    d.mkdir(parents=True, exist_ok=True)
    for fname, url, size, sha in spec["files"]:
        dst = d / fname
        if dst.exists() and sha256(dst) == sha:
            print(f"  {fname}: present, SHA-256 ok")
            continue
        if copy_from:
            src = Path(copy_from) / fname
            print(f"  {fname}: copying from {src}")
            shutil.copyfile(src, dst)
        else:
            print(
                f"  {fname}: downloading {url}" + (f" ({size / 1e6:.1f} MB)" if size else ""),
                flush=True,
            )
            download(url, dst)
        got = sha256(dst)
        if got != sha:
            dst.unlink()
            raise SystemExit(f"  {fname}: SHA-256 {got} does not match {sha}; deleted")
        print(f"  {fname}: SHA-256 ok")
    for fname, sha in spec.get("unzip", {}).items():
        dst = d / fname
        if not (dst.exists() and sha256(dst) == sha):
            with zipfile.ZipFile(d / spec["files"][0][0]) as z:
                member = next(m for m in z.namelist() if m.endswith(fname))
                dst.write_bytes(z.read(member))
            if sha256(dst) != sha:
                raise SystemExit(f"  {fname}: SHA-256 mismatch after unzipping")
        print(f"  {fname}: unzipped, SHA-256 ok")
    with open(d / f"SOURCES-{name}.md", "w") as f:
        f.write(
            f"## {name}\n\n{spec['what']}\n\n- Source: {spec['page']}\n- License: "
            f"{spec['license']}\n"
            f"- Cite: {spec['cite']}\n- Files: " + ", ".join(x[0] for x in spec["files"]) + "\n\n"
        )
    if "frostt" in spec and spec["frostt"] != "lbnl-network":
        sys.path.insert(0, str(Path(__file__).resolve().parent / "harness"))
        import datasets as D

        if not Path(D.FROSTT, f"{spec['frostt']}.npz").exists():
            print(f"  converting to {D.FROSTT}/{spec['frostt']}.npz", flush=True)
            D.convert_frostt(spec["frostt"])
    print(f"  license: {spec['license']}\n  cite: {spec['cite']}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("names", nargs="*", help="datasets, or 'all'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument(
        "--copy-from",
        default=None,
        help="take the files from this directory instead of downloading",
    )
    args = ap.parse_args()
    paths.set_data_dir(args.data_dir)
    if args.list or not args.names:
        print(f"data directory: {paths.data_dir()}\n")
        for name, spec in DATASETS.items():
            size = sum(x[2] or 0 for x in spec["files"])
            print(
                f"{name:15s} {size / 1e6:8.1f} MB  {spec['what']}\n{'':26s}license: "
                f"{spec['license']}"
            )
        return
    names = list(DATASETS) if args.names == ["all"] else args.names
    for name in names:
        if name not in DATASETS:
            raise SystemExit(f"unknown dataset {name!r}; see --list")
    for name in names:
        print(f"{name} -> {paths.data_dir() / DATASETS[name]['dir']}", flush=True)
        fetch(name, args.copy_from)


if __name__ == "__main__":
    main()
