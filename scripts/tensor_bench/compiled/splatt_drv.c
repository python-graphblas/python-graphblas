/* SPLATT MTTKRP driver: SPLATT's own CSF and mttkrp_csf (as `splatt cpd` calls it).
 *
 *   splatt_drv TENSOR.tb FACTORS.f64 R ALLOC NITER_MAX THREADS...
 *
 * ALLOC is onemode, twomode (SPLATT's default for cpd) or allmode. Reads the binary tensor
 * and factors that export_bin.py writes, so outputs are comparable with the Python libraries.
 * Empty slices are kept (splatt_csf_convert would compact them and renumber the rows), so
 * output row i is index i. Prints one JSON line per (threads, mode) with the best and median
 * time, and a result fingerprint computed as harness/common.py does.
 */
#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "base.h"
#include "csf.h"
#include "matrix.h"
#include "mttkrp.h"
#include "sptensor.h"
#include "thd_info.h"

static uint64_t mix(uint64_t lin)
{
  uint64_t h = lin * 0x9E3779B97F4A7C15ULL;
  h ^= h >> 29;
  h *= 0xBF58476D1CE4E5B9ULL;
  h ^= h >> 32;
  return h;
}

static void fingerprint_dense(val_t const *a, uint64_t I, uint64_t R, char *buf, size_t n)
{
  uint64_t count = 0, hs = 0;
  double vs = 0, ws = 0;
  for (uint64_t x = 0; x < I * R; ++x) {
    if (a[x] == 0) continue;
    uint64_t h = mix(x);
    count++;
    hs += h;
    vs += a[x];
    ws += a[x] * ((double)(h >> 11) * 0x1p-53);
  }
  snprintf(buf, n,
           "{\"shape\": [%" PRIu64 ", %" PRIu64 "], \"count\": %" PRIu64
           ", \"hsum\": \"0x%" PRIx64 "\", \"vsum\": %.17g, \"wsum\": %.17g, \"sorted\": true}",
           I, R, count, hs, vs, ws);
}

static int cmp_double(void const *a, void const *b)
{
  double x = *(double const *)a, y = *(double const *)b;
  return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
  if (argc < 7) {
    fprintf(stderr, "usage: %s TENSOR.tb FACTORS.f64 R ALLOC NITER_MAX THREADS...\n", argv[0]);
    return 2;
  }
  char const *tfile = argv[1], *ffile = argv[2], *alloc = argv[4];
  idx_t const R = (idx_t)atoll(argv[3]);
  int const nmax = atoi(argv[5]);

  /* read the tensor */
  FILE *f = fopen(tfile, "rb");
  char magic[8];
  int64_t hdr[2];
  if (!f || fread(magic, 1, 8, f) != 8 || memcmp(magic, "TNSB0001", 8) || fread(hdr, 8, 2, f) != 2) {
    fprintf(stderr, "bad tensor file %s\n", tfile);
    return 1;
  }
  idx_t const nmodes = (idx_t)hdr[0], nnz = (idx_t)hdr[1];
  int64_t dims[MAX_NMODES];
  if (fread(dims, 8, nmodes, f) != nmodes) return 1;
  idx_t **inds = malloc(nmodes * sizeof(*inds));
  for (idx_t m = 0; m < nmodes; ++m) {
    inds[m] = splatt_malloc(nnz * sizeof(idx_t));
    if (fread(inds[m], 8, nnz, f) != nnz) return 1;
  }
  val_t *vals = splatt_malloc(nnz * sizeof(val_t));
  if (fread(vals, 8, nnz, f) != nnz) return 1;
  fclose(f);

  /* factors: mode after mode, I_m x R row-major */
  matrix_t *mats[MAX_NMODES + 1];
  f = fopen(ffile, "rb");
  idx_t maxdim = 0;
  for (idx_t m = 0; m < nmodes; ++m) {
    mats[m] = mat_alloc(dims[m], R);
    if (fread(mats[m]->vals, 8, dims[m] * R, f) != (size_t)(dims[m] * R)) {
      fprintf(stderr, "bad factor file %s\n", ffile);
      return 1;
    }
    if ((idx_t)dims[m] > maxdim) maxdim = dims[m];
  }
  fclose(f);
  mats[MAX_NMODES] = mat_alloc(maxdim, R);

  double *opts = splatt_default_opts();
  opts[SPLATT_OPTION_VERBOSITY] = SPLATT_VERBOSITY_NONE;
  if (!strcmp(alloc, "onemode"))
    opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
  else if (!strcmp(alloc, "twomode"))
    opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_TWOMODE;
  else if (!strcmp(alloc, "allmode"))
    opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ALLMODE;
  else {
    fprintf(stderr, "unknown alloc %s\n", alloc);
    return 2;
  }
  int maxthreads = 1;
  for (int a = 6; a < argc; ++a)
    if (atoi(argv[a]) > maxthreads) maxthreads = atoi(argv[a]);
  opts[SPLATT_OPTION_NTHREADS] = maxthreads;
  omp_set_num_threads(maxthreads);

  /* CSF build: what splatt_csf_convert does, minus tt_remove_empty */
  double t0 = omp_get_wtime();
  sptensor_t tt;
  tt_fill(&tt, nnz, nmodes, inds, vals);
  for (idx_t m = 0; m < nmodes; ++m) tt.dims[m] = dims[m];
  splatt_csf *cs = csf_alloc(&tt, opts);
  double csf_s = omp_get_wtime() - t0;
  unsigned long csf_bytes = csf_storage(cs, opts);

  char fp[512];
  for (int a = 6; a < argc; ++a) {
    int const nt = atoi(argv[a]);
    opts[SPLATT_OPTION_NTHREADS] = nt;
    omp_set_num_threads(nt);
    t0 = omp_get_wtime();
    thd_info *thds = thd_init(nt, 3, (nmodes * R * sizeof(val_t)) + 64, 0, (nmodes * R * sizeof(val_t)) + 64);
    splatt_mttkrp_ws *ws = splatt_mttkrp_alloc_ws(cs, R, opts);
    double ws_s = omp_get_wtime() - t0;
    for (idx_t m = 0; m < nmodes; ++m) {
      double times[64];
      t0 = omp_get_wtime();
      mttkrp_csf(cs, mats, m, thds, ws, opts); /* warm-up */
      double first = omp_get_wtime() - t0;
      int n = first < 0.1 ? 9 : first < 1 ? 5 : 3;
      if (n > nmax) n = nmax;
      for (int it = 0; it < n; ++it) {
        t0 = omp_get_wtime();
        mttkrp_csf(cs, mats, m, thds, ws, opts);
        times[it] = omp_get_wtime() - t0;
      }
      fingerprint_dense(mats[MAX_NMODES]->vals, dims[m], R, fp, sizeof fp);
      qsort(times, n, sizeof(double), cmp_double);
      printf("{\"lib\": \"splatt\", \"alloc\": \"%s\", \"case\": \"mttkrp%" PRIu64 "_r%" PRIu64
             "\", \"nthreads\": %d, \"best_ms\": %.6f, \"median_ms\": %.6f, \"n\": %d, \"first_ms\": %.4f"
             ", \"prep_ms\": %.3f, \"ws_ms\": %.3f, \"csf_bytes\": %lu, \"num_csf\": %" PRIu64
             ", \"privatized\": %d, \"fp\": %s}\n",
             alloc, (uint64_t)m, (uint64_t)R, nt, 1e3 * times[0], 1e3 * times[n / 2], n, 1e3 * first,
             1e3 * csf_s, 1e3 * ws_s, csf_bytes, (uint64_t)ws->num_csf, (int)ws->is_privatized[m], fp);
      fflush(stdout);
    }
    splatt_mttkrp_free_ws(ws);
    thd_free(thds, nt);
  }
  csf_free(cs, opts);
  return 0;
}
