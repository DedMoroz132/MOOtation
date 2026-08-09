/* SPDX-License-Identifier: Apache-2.0
 * ===========================================================================
 * MOOtation — C ABI.
 *
 * This is the bridge for everything that is not C++: Python (ctypes/cffi),
 * C#, Rust, Julia, Fortran, MATLAB, Go, LabVIEW, anything with an FFI.
 *
 * Nothing but C types crosses the boundary. Configuration goes in as one
 * string in the `key = value` format of mootation/settings.hpp; candidates and
 * objective values move as flat row-major double arrays. There is no struct to
 * keep in sync between languages, which is the whole point.
 *
 * THE LOOP
 *
 *   moo_session* s = moo_open(settings_text);
 *   int nv = moo_n_vars(s), no = moo_n_objs(s);
 *
 *   int n = moo_ask_count(s);              // candidates waiting, 0 = finished
 *   while (n > 0) {
 *       double* x = malloc(n * nv * sizeof(double));
 *       double* f = malloc(n * no * sizeof(double));
 *       moo_ask(s, x, n * nv);             // x is n rows of nv variables
 *       for (int i = 0; i < n; ++i)
 *           my_simulator(&x[i * nv], &f[i * no]);
 *       n = moo_tell(s, f, n * no, NULL, 0);   // returns the NEXT count
 *       free(x); free(f);
 *   }
 *
 *   int m = moo_result_count(s);           // the final population
 *   ...
 *   moo_close(s);
 *
 * THE COUNT IS THE ALGORITHM'S CHOICE, not pop_size. Generational algorithms
 * hand over a whole offspring generation; steady-state ones (MOEA/D-DE,
 * MOEA/DD, ...) hand over one candidate at a time, because that is how those
 * algorithms are defined. Always re-read moo_ask_count / the moo_tell return
 * value; never assume.
 *
 * ERRORS
 *   Every function that can fail returns a negative int (or NULL) and leaves a
 *   message in moo_last_error(). The message belongs to the session and stays
 *   valid until the next call on that session. For moo_open, which has no
 *   session yet, pass NULL to moo_last_error to read the last open failure —
 *   that one is thread-local.
 *
 * THREADS
 *   Sessions are independent; drive each one from a single thread at a time.
 * ===========================================================================
 */

#ifndef MOOTATION_CAPI_H
#define MOOTATION_CAPI_H

#if defined(_WIN32)
#  if defined(MOOTATION_C_BUILD)
#    define MOO_API __declspec(dllexport)
#  else
#    define MOO_API __declspec(dllimport)
#  endif
#else
#  define MOO_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct moo_session moo_session;

/* ── Library-level, no session needed ─────────────────────────────────────── */

/* Encoded as major*10000 + minor*100 + patch. */
MOO_API int         moo_version(void);
MOO_API const char* moo_version_string(void);

/* The names accepted as `algorithm = ...` in the settings text. */
MOO_API int         moo_algorithm_count(void);
/* NULL if i is out of range. The pointer is static; do not free it. */
MOO_API const char* moo_algorithm_name(int i);

/* ── Opening and closing ──────────────────────────────────────────────────── */

/* `text` is the settings format of mootation/settings.hpp — the same text a
 * config file holds. Returns NULL on error; see moo_last_error(NULL). */
MOO_API moo_session* moo_open(const char* text);
/* Reads the file and calls moo_open on its contents. */
MOO_API moo_session* moo_open_file(const char* path);
/* Safe on NULL. Stops the run if it is still in progress. */
MOO_API void         moo_close(moo_session* s);

/* ── Shape ────────────────────────────────────────────────────────────────── */
MOO_API int moo_n_vars(const moo_session* s);
MOO_API int moo_n_objs(const moo_session* s);
MOO_API int moo_n_cons(const moo_session* s);
/* Generations completed so far. */
MOO_API int moo_generation(const moo_session* s);

/* ── The loop ─────────────────────────────────────────────────────────────── */

/* How many candidates are waiting to be evaluated. 0 means the run is over;
 * negative means error. */
MOO_API int moo_ask_count(moo_session* s);

/* Copies the waiting candidates into `x` as `count` rows of `moo_n_vars()`
 * doubles. `x_len` is the number of doubles `x` can hold and is checked.
 * Returns the number of candidates written, or negative on error. */
MOO_API int moo_ask(moo_session* s, double* x, int x_len);

/* Posts objective values for the batch moo_ask returned and advances the run.
 *   f     : `count` rows of moo_n_objs() doubles, row-major.
 *   g     : `count` rows of moo_n_cons() doubles, row-major, each <= 0 meaning
 *           the constraint is satisfied. Pass NULL and g_len = 0 when
 *           moo_n_cons() is 0.
 * Returns the number of candidates in the NEXT batch — 0 when the run has
 * finished — or negative on error. */
MOO_API int moo_tell(moo_session* s,
                     const double* f, int f_len,
                     const double* g, int g_len);

/* ── The answer ───────────────────────────────────────────────────────────── */

/* Size of the final population. Blocks until the run ends. Negative on error;
 * it is an error to call this while a batch is still waiting for moo_tell. */
MOO_API int moo_result_count(moo_session* s);

/* Copies the final population out. Any of the three pointers may be NULL to
 * skip that part; the matching length must then be 0.
 *   x  : count rows of moo_n_vars() doubles
 *   f  : count rows of moo_n_objs() doubles
 *   cv : count doubles, the total constraint violation (0 = feasible)
 * Returns the number of individuals written, or negative on error. */
MOO_API int moo_result(moo_session* s,
                       double* x,  int x_len,
                       double* f,  int f_len,
                       double* cv, int cv_len);

/* Knobs that were set but the chosen algorithm does not have. They are
 * reported rather than dropped: a run configured with an ignored parameter is
 * not the run that was asked for. Available after the run finishes. */
MOO_API int         moo_ignored_count(moo_session* s);
MOO_API const char* moo_ignored_name(moo_session* s, int i);

/* ── Errors ───────────────────────────────────────────────────────────────── */

/* The last error on `s`, or the last moo_open failure when `s` is NULL.
 * Returns NULL when there is none. Valid until the next call on that session
 * (or, for the NULL form, until the next moo_open on this thread). */
MOO_API const char* moo_last_error(const moo_session* s);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MOOTATION_CAPI_H */
