// TACO driver: TACO-generated kernels (C++ API, default schedule, OpenMP).
//
//   taco_drv PREFIX GROUP NITER_MAX THREADS...
//
// GROUP: mttkrp16, mttkrp64, ttv, ttm16, ewise, sum. Reads PREFIX.tb (and PREFIX_y.tb,
// PREFIX_f<R>.f64, PREFIX_v.f64), as export_bin.py writes them. For each mode the input is
// stored as a CSF (all levels compressed) in the mode order the kernel wants: the output mode
// at the root for MTTKRP (so the default schedule parallelizes the outer loop without races),
// the contracted or summed mode at the leaf for TTV, TTM and sums, the identity order for
// element-wise ops. The CSF arrays are built here from the sorted coordinates (prep_ms);
// TACO's compile is reported as compile_ms. One JSON line per (case, threads): best and median
// of assemble + compute, and of compute alone, with a fingerprint as harness/common.py
// computes it.
//
// TACO's command-line tool cannot be used instead: on master (0e79acb5) its benchmark path
// compiles a module whose C source defines `compute` twice ("redefinition of 'compute'").
// Note also that TensorBase::assemble() and compute() return at once unless the tensor is
// flagged as needing them, so repeated timing must set the flags (time_kernel below).
#include <taco.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

using namespace taco;
using std::string;
using std::vector;

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Coo {
  int nmodes;
  int64_t nnz;
  vector<int64_t> dims;
  vector<vector<int64_t>> ind;
  vector<double> vals;
};

static Coo read_tb(const string& path) {
  Coo t;
  FILE* f = fopen(path.c_str(), "rb");
  char magic[8];
  int64_t hdr[2];
  if (!f || fread(magic, 1, 8, f) != 8 || memcmp(magic, "TNSB0001", 8) || fread(hdr, 8, 2, f) != 2) {
    fprintf(stderr, "bad tensor file %s\n", path.c_str());
    exit(1);
  }
  t.nmodes = (int)hdr[0];
  t.nnz = hdr[1];
  t.dims.resize(t.nmodes);
  if (fread(t.dims.data(), 8, t.nmodes, f) != (size_t)t.nmodes) exit(1);
  t.ind.resize(t.nmodes);
  for (int m = 0; m < t.nmodes; ++m) {
    t.ind[m].resize(t.nnz);
    if (fread(t.ind[m].data(), 8, t.nnz, f) != (size_t)t.nnz) exit(1);
  }
  t.vals.resize(t.nnz);
  if (fread(t.vals.data(), 8, t.nnz, f) != (size_t)t.nnz) exit(1);
  fclose(f);
  return t;
}

static vector<double> read_f64(const string& path, size_t n) {
  vector<double> v(n);
  FILE* f = fopen(path.c_str(), "rb");
  if (!f || fread(v.data(), 8, n, f) != n) {
    fprintf(stderr, "bad file %s\n", path.c_str());
    exit(1);
  }
  fclose(f);
  return v;
}

// A CSF (every level compressed) TACO tensor of t in mode order `order`.
static TensorBase make_csf(const string& name, const Coo& t, const vector<int>& order) {
  int n = t.nmodes;
  vector<int64_t> perm(t.nnz);
  std::iota(perm.begin(), perm.end(), 0);
  bool identity = true;
  for (int l = 0; l < n; ++l) identity &= order[l] == l;
  if (!identity) {  // input is C-order sorted already
    vector<uint64_t> key(t.nnz);
    for (int64_t e = 0; e < t.nnz; ++e) {
      uint64_t k = 0;
      for (int l = 0; l < n; ++l) k = k * (uint64_t)t.dims[order[l]] + (uint64_t)t.ind[order[l]][e];
      key[e] = k;
    }
    std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) { return key[a] < key[b]; });
  }
  vector<vector<int>> pos(n), crd(n);
  vector<double> vals(t.nnz);
  for (int64_t q = 0; q < t.nnz; ++q) {
    int64_t e = perm[q];
    int l = 0;
    if (q > 0) {
      int64_t p = perm[q - 1];
      while (l < n && t.ind[order[l]][e] == t.ind[order[l]][p]) ++l;
    }
    for (int lev = l; lev < n; ++lev) {
      crd[lev].push_back((int)t.ind[order[lev]][e]);
      if (lev + 1 < n) pos[lev + 1].push_back((int)crd[lev + 1].size());
    }
    vals[q] = t.vals[e];
  }
  pos[0] = {0, (int)crd[0].size()};
  for (int lev = 1; lev < n; ++lev) pos[lev].push_back((int)crd[lev].size());
  vector<int> dims(n);
  for (int m = 0; m < n; ++m) dims[m] = (int)t.dims[m];
  Format fmt(vector<ModeFormatPack>(n, Sparse), order);
  Tensor<double> T(name, dims, fmt);
  auto storage = T.getStorage();
  vector<ModeIndex> mis;
  for (int lev = 0; lev < n; ++lev) mis.push_back(ModeIndex({makeArray(pos[lev]), makeArray(crd[lev])}));
  storage.setIndex(Index(fmt, mis));
  storage.setValues(makeArray(vals));
  T.setStorage(storage);
  return T;
}

static TensorBase make_dense(const string& name, const vector<int>& dims, const double* data) {
  size_t n = 1;
  for (int d : dims) n *= d;
  Format fmt(vector<ModeFormatPack>(dims.size(), Dense));
  Tensor<double> T(name, dims, fmt);
  auto storage = T.getStorage();
  vector<ModeIndex> mis;
  for (int d : dims) mis.push_back(ModeIndex({makeArray({d})}));
  storage.setIndex(Index(fmt, mis));
  storage.setValues(makeArray(vector<double>(data, data + n)));
  T.setStorage(storage);
  return T;
}

// Fingerprint (harness/common.py) of a TACO tensor in natural mode order.
struct FP {
  uint64_t count = 0, h = 0;
  double v = 0, w = 0;
  void add(uint64_t lin, double val) {
    if (val == 0) return;
    uint64_t x = lin * 0x9E3779B97F4A7C15ULL;
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 32;
    count++;
    h += x;
    v += val;
    w += val * ((double)(x >> 11) * 0x1p-53);
  }
};

static void walk(const TensorBase& T, int lev, int64_t ppos, uint64_t prefix, const double* vals, FP& fp) {
  auto fmt = T.getFormat();
  int n = T.getOrder();
  if (lev == n) {
    fp.add(prefix, vals[ppos]);
    return;
  }
  int64_t dim = T.getDimension(lev);
  auto mi = T.getStorage().getIndex().getModeIndex(lev);
  if (fmt.getModeFormats()[lev] == Dense) {
    for (int64_t c = 0; c < dim; ++c) walk(T, lev + 1, ppos * dim + c, prefix * dim + c, vals, fp);
  } else {
    const int* pos = (const int*)mi.getIndexArray(0).getData();
    const int* crd = (const int*)mi.getIndexArray(1).getData();
    for (int64_t q = pos[ppos]; q < pos[ppos + 1]; ++q) walk(T, lev + 1, q, prefix * dim + crd[q], vals, fp);
  }
}

static string fingerprint(const TensorBase& T) {
  FP fp;
  walk(T, 0, 0, 0, (const double*)T.getStorage().getValues().getData(), fp);
  string shape = "[";
  for (int m = 0; m < T.getOrder(); ++m) shape += (m ? ", " : "") + std::to_string(T.getDimension(m));
  shape += "]";
  char buf[512];
  snprintf(buf, sizeof buf,
           "{\"shape\": %s, \"count\": %" PRIu64 ", \"hsum\": \"0x%" PRIx64
           "\", \"vsum\": %.17g, \"wsum\": %.17g, \"sorted\": true}",
           shape.c_str(), fp.count, fp.h, fp.v, fp.w);
  return buf;
}

static int g_nmax = 9;

struct Times {
  double best, median, cbest, cmedian, first;
  int n;
};

// Time assemble + compute (and compute alone): one warm-up, then best of N.
static Times time_kernel(TensorBase& A, bool sparse_out) {
  Times r;
  double t0 = now();
  A.assemble();
  A.compute();
  r.first = now() - t0;
  int n = r.first < 0.1 ? 9 : r.first < 1 ? 5 : 3;
  n = std::min(n, g_nmax);
  vector<double> both, comp;
  for (int it = 0; it < n; ++it) {
    // assemble() and compute() return at once unless flagged (TensorBase::compute: needsCompute)
    t0 = now();
    if (sparse_out) {
      A.setNeedsAssemble(true);
      A.assemble();
    }
    double t1 = now();
    A.setNeedsCompute(true);
    A.compute();
    double t2 = now();
    both.push_back(t2 - t0);
    comp.push_back(t2 - t1);
  }
  std::sort(both.begin(), both.end());
  std::sort(comp.begin(), comp.end());
  r.best = both[0];
  r.median = both[n / 2];
  r.cbest = comp[0];
  r.cmedian = comp[n / 2];
  r.n = n;
  return r;
}

// TACO_DRV_SRC=<dir>: write each generated kernel's C source there (to check for OpenMP pragmas)
static void dump_source(TensorBase& A, const string& kase) {
  const char* dir = getenv("TACO_DRV_SRC");
  if (!dir) return;
  FILE* f = fopen((string(dir) + "/" + kase + ".c").c_str(), "w");
  if (f) {
    fputs(A.getSource().c_str(), f);
    fclose(f);
  }
}

static void emit(const string& kase, int nt, const Times& t, double prep, double compile, const string& fp,
                 const string& extra = "") {
  printf("{\"lib\": \"taco\", \"drv\": 2, \"case\": \"%s\", \"nthreads\": %d, \"best_ms\": %.6f, \"median_ms\": %.6f, "
         "\"compute_best_ms\": %.6f, \"compute_median_ms\": %.6f, \"n\": %d, \"first_ms\": %.4f, "
         "\"prep_ms\": %.3f, \"compile_ms\": %.3f%s, \"fp\": %s}\n",
         kase.c_str(), nt, 1e3 * t.best, 1e3 * t.median, 1e3 * t.cbest, 1e3 * t.cmedian, t.n, 1e3 * t.first,
         1e3 * prep, 1e3 * compile, extra.c_str(), fp.c_str());
  fflush(stdout);
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s PREFIX GROUP NITER_MAX THREADS...\n", argv[0]);
    return 2;
  }
  string ds = argv[1], group = argv[2];
  g_nmax = atoi(argv[3]);
  vector<int> threads;
  for (int a = 4; a < argc; ++a) threads.push_back(atoi(argv[a]));
  string bin = ds;  // the path prefix of the exported files
  taco_set_parallel_schedule(ParallelSchedule::Dynamic, 16);

  Coo X = read_tb(bin + ".tb");
  int n = X.nmodes;
  vector<int> dims(n);
  for (int m = 0; m < n; ++m) dims[m] = (int)X.dims[m];
  vector<IndexVar> iv(n);
  IndexVar l("l");

  if (group == "mttkrp16" || group == "mttkrp64" || group == "ttm16") {
    int R = group == "mttkrp64" ? 64 : 16;
    size_t total = 0;
    for (int d : dims) total += (size_t)d * R;
    vector<double> F = read_f64(bin + "_f" + std::to_string(R) + ".f64", total);
    vector<TensorBase> Fs;
    size_t off = 0;
    for (int m = 0; m < n; ++m) {
      Fs.push_back(make_dense("F" + std::to_string(m), {dims[m], R}, F.data() + off));
      off += (size_t)dims[m] * R;
    }
    bool ttm = group == "ttm16";
    for (int m = 0; m < n; ++m) {
      vector<int> order;
      if (ttm) {
        for (int a = 0; a < n; ++a)
          if (a != m) order.push_back(a);
        order.push_back(m);
      } else {
        order.push_back(m);
        for (int a = 0; a < n; ++a)
          if (a != m) order.push_back(a);
      }
      double t0 = now();
      TensorBase B = make_csf("B", X, order);
      double prep = now() - t0;
      TensorBase A;
      if (ttm) {
        vector<int> odims;
        vector<IndexVar> ov;
        for (int a = 0; a < n; ++a)
          if (a != m) odims.push_back(dims[a]), ov.push_back(iv[a]);
        odims.push_back(R);
        ov.push_back(l);
        vector<ModeFormatPack> mf(n - 1, Sparse);
        mf.push_back(Dense);
        A = Tensor<double>("A", odims, Format(mf));
        vector<IndexVar> bv(iv.begin(), iv.end());
        A(ov) = B(bv) * Fs[m](iv[m], l);
      } else {
        A = Tensor<double>("A", {dims[m], R}, Format({Dense, Dense}));
        IndexExpr e = B(vector<IndexVar>(iv.begin(), iv.end()));
        for (int a = 0; a < n; ++a)
          if (a != m) e = e * Fs[a](iv[a], l);
        A(iv[m], l) = e;
      }
      t0 = now();
      A.compile();
      double compile = now() - t0;
      string kase = (ttm ? "ttm" : "mttkrp") + std::to_string(m) + "_r" + std::to_string(R);
      dump_source(A, kase);
      for (int nt : threads) {
        taco_set_num_threads(nt);
        Times t = time_kernel(A, ttm);
        emit(kase, nt, t, prep, compile, fingerprint(A));
      }
    }
  } else if (group == "ttv" || group == "sum") {
    bool ttv = group == "ttv";
    vector<double> V = read_f64(bin + "_v.f64", std::accumulate(dims.begin(), dims.end(), (size_t)0));
    size_t off = 0;
    for (int m = 0; m < n; ++m) {
      vector<int> order, odims;
      vector<IndexVar> ov;
      for (int a = 0; a < n; ++a)
        if (a != m) order.push_back(a), odims.push_back(dims[a]), ov.push_back(iv[a]);
      order.push_back(m);
      double t0 = now();
      TensorBase B = make_csf("B", X, order);
      double prep = now() - t0;
      TensorBase v = make_dense("v", {dims[m]}, V.data() + off);
      off += dims[m];
      TensorBase A = Tensor<double>("A", odims, Format(vector<ModeFormatPack>(n - 1, Sparse)));
      vector<IndexVar> bv(iv.begin(), iv.end());
      if (ttv)
        A(ov) = B(bv) * v(iv[m]);
      else
        A(ov) = B(bv);
      t0 = now();
      A.compile();
      double compile = now() - t0;
      string kase = (ttv ? "ttv" : "sum") + std::to_string(m);
      dump_source(A, kase);
      for (int nt : threads) {
        taco_set_num_threads(nt);
        Times t = time_kernel(A, true);
        emit(kase, nt, t, prep, compile, fingerprint(A));
      }
    }
  } else if (group == "ewise") {
    Coo Y = read_tb(bin + "_y.tb");
    vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    double t0 = now();
    TensorBase B = make_csf("B", X, order);
    TensorBase C = make_csf("C", Y, order);
    double prep = now() - t0;
    for (string op : {"add_same", "mul_same"}) {
      TensorBase A = Tensor<double>("A", dims, Format(vector<ModeFormatPack>(n, Sparse)));
      vector<IndexVar> bv(iv.begin(), iv.end());
      if (op == "add_same")
        A(bv) = B(bv) + C(bv);
      else
        A(bv) = B(bv) * C(bv);
      t0 = now();
      A.compile();
      double compile = now() - t0;
      dump_source(A, op);
      for (int nt : threads) {
        taco_set_num_threads(nt);
        Times t = time_kernel(A, true);
        emit(op, nt, t, prep, compile, fingerprint(A));
      }
    }
  } else {
    fprintf(stderr, "unknown group %s\n", group.c_str());
    return 2;
  }
  return 0;
}
