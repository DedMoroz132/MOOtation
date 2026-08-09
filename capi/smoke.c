/* SPDX-License-Identifier: Apache-2.0
 *
 * A C consumer of the C ABI, compiled as C99. Its job is twofold: prove the
 * ask/tell loop works through the ABI, and prove capi.h is really C — a
 * `std::` or a default argument that crept into the header fails here rather
 * than in someone else's project.
 *
 * The "simulator" is ZDT1 written out by hand. Nothing below ever tells
 * MOOtation what the problem is; it only ever sees numbers.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mootation/capi.h"

#define N_VARS 10
#define N_OBJS 2

static void simulate(const double* x, double* f) {
    double g = 0.0;
    int i;
    for (i = 1; i < N_VARS; ++i) g += x[i];
    g = 1.0 + 9.0 * g / (double)(N_VARS - 1);
    f[0] = x[0];
    f[1] = g * (1.0 - sqrt(f[0] / g));
}

static int fail(const char* what, moo_session* s) {
    const char* e = moo_last_error(s);
    fprintf(stderr, "FAILED: %s%s%s\n", what, e ? " — " : "", e ? e : "");
    if (s) moo_close(s);
    return 1;
}

int main(void) {
    static const char* kSettings =
        "algorithm = nsga2\n"
        "pop_size  = 40\n"
        "max_gen   = 40\n"
        "seed      = 7\n"
        "n_vars    = 10\n"
        "n_objs    = 2\n"
        "lower     = 0\n"
        "upper     = 1\n";

    moo_session* s;
    int n, nv, no, i, batches = 0, evaluated = 0, count;
    double *x = NULL, *f = NULL, *rx = NULL, *rf = NULL, *rcv = NULL;
    double best = 1e300;

    printf("MOOtation %s, %d algorithms\n", moo_version_string(),
           moo_algorithm_count());
    if (moo_algorithm_count() != 60) return fail("expected 60 algorithms", NULL);
    if (moo_algorithm_name(0) == NULL) return fail("algorithm_name(0)", NULL);
    if (moo_algorithm_name(-1) != NULL || moo_algorithm_name(99999) != NULL)
        return fail("algorithm_name must reject an out-of-range index", NULL);

    /* A bad configuration must come back as an error, not a crash. */
    if (moo_open("algorithm = no_such_algorithm\nn_vars = 2\nlower = 0\nupper = 1\n") != NULL)
        return fail("an unknown algorithm should not open", NULL);
    if (moo_open("this is not a settings file") != NULL)
        return fail("malformed settings should not open", NULL);
    if (moo_last_error(NULL) == NULL)
        return fail("a failed open must leave a message", NULL);

    s = moo_open(kSettings);
    if (!s) return fail("moo_open", NULL);

    nv = moo_n_vars(s);
    no = moo_n_objs(s);
    if (nv != N_VARS || no != N_OBJS) return fail("unexpected shape", s);

    n = moo_ask_count(s);
    if (n < 0) return fail("moo_ask_count", s);

    x = (double*)malloc((size_t)n * (size_t)nv * sizeof(double));
    f = (double*)malloc((size_t)n * (size_t)no * sizeof(double));
    if (!x || !f) return fail("out of memory", s);

    while (n > 0) {
        /* The batch size is the algorithm's choice, so the buffers grow to
         * whatever the current batch needs rather than being sized once. */
        double* nx = (double*)realloc(x, (size_t)n * (size_t)nv * sizeof(double));
        double* nf = (double*)realloc(f, (size_t)n * (size_t)no * sizeof(double));
        if (!nx || !nf) return fail("out of memory", s);
        x = nx; f = nf;

        if (moo_ask(s, x, n * nv) != n) return fail("moo_ask", s);
        for (i = 0; i < n; ++i) simulate(&x[i * nv], &f[i * no]);

        evaluated += n;              /* count what was just evaluated ... */
        n = moo_tell(s, f, n * no, NULL, 0);   /* ... before n becomes the NEXT batch */
        if (n < 0) return fail("moo_tell", s);
        ++batches;
    }

    count = moo_result_count(s);
    if (count != 40) return fail("expected 40 solutions", s);

    rx  = (double*)malloc((size_t)count * (size_t)nv * sizeof(double));
    rf  = (double*)malloc((size_t)count * (size_t)no * sizeof(double));
    rcv = (double*)malloc((size_t)count * sizeof(double));
    if (!rx || !rf || !rcv) return fail("out of memory", s);

    /* A short buffer must be rejected, not overrun. The check is on moo_result
     * rather than moo_ask because by now the run has finished and moo_ask
     * correctly reports "nothing waiting" rather than an error, which would
     * make the assertion vacuous. */
    if (moo_result(s, NULL, 0, NULL, 0, rcv, count - 1) >= 0)
        return fail("a short cv buffer should have been rejected", s);
    if (moo_last_error(s) == NULL)
        return fail("a rejected call must leave a message", s);

    if (moo_result(s, rx, count * nv, rf, count * no, rcv, count) != count)
        return fail("moo_result", s);

    for (i = 0; i < count; ++i) {
        double sum = rf[i * no] + rf[i * no + 1];
        if (sum < best) best = sum;
    }

    printf("c api: %d batches, %d evaluations, %d generations, %d solutions\n",
           batches, evaluated, moo_generation(s), count);
    printf("best f1+f2 = %.6f (analytic minimum on the ZDT1 front is 0.75)\n", best);

    /* ZDT1: f2 = 1 - sqrt(f1), so f1 + f2 bottoms out at 3/4. */
    if (best < 0.75 - 1e-9) return fail("a solution beat the true front", s);
    /* 40 individuals for 40 generations is a smoke test, not a budget: it lands
       around 0.80. A random population sits near 4-5, so this still separates
       "the optimizer ran" from "the optimizer did nothing". */
    if (best > 0.85)        return fail("the run did not converge", s);

    free(x); free(f); free(rx); free(rf); free(rcv);
    moo_close(s);
    moo_close(NULL);   /* must be safe */

    printf("OK\n");
    return 0;
}
