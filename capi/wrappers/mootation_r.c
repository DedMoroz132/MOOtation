/* SPDX-License-Identifier: Apache-2.0
 *
 * The R shim.
 *
 * R cannot call the C ABI directly. `.C()` marshals only atomic vectors, so an
 * opaque `moo_session*` cannot survive a round trip through it, and `.Call()`
 * requires functions with the R API signature (SEXP in, SEXP out) rather than
 * plain C ones. This file is the missing adapter: it holds the session in an
 * R external pointer and converts vectors on both sides.
 *
 * Build it next to the shared library:
 *
 *     R CMD SHLIB mootation_r.c -L<dir with libmootation> -lmootation \
 *         -I<repo>/include
 *
 * then from R:
 *
 *     dyn.load("mootation_r.so")     # or .dll
 *     source("mootation.R")
 *
 * NOT COMPILED OR RUN ON THE MACHINE THIS WAS WRITTEN ON — no R was installed.
 * The C ABI it calls is tested (capi/smoke.c, capi/ctypes_demo.py); this
 * adapter is not. Treat it as a starting point, verify it before relying on it,
 * and please report what needed changing.
 */

#include <R.h>
#include <Rinternals.h>

#include "mootation/capi.h"

static void moo_r_finalizer(SEXP ext)
{
    moo_session *s = (moo_session *)R_ExternalPtrAddr(ext);
    if (s) {
        moo_close(s);
        R_ClearExternalPtr(ext);
    }
}

static moo_session *moo_r_handle(SEXP ext)
{
    moo_session *s = (moo_session *)R_ExternalPtrAddr(ext);
    if (!s) error("mootation: the session has already been closed");
    return s;
}

SEXP R_moo_version(void)
{
    return mkString(moo_version_string());
}

SEXP R_moo_algorithms(void)
{
    int n = moo_algorithm_count();
    if (n < 0) error("mootation: moo_algorithm_count failed");
    SEXP out = PROTECT(allocVector(STRSXP, n));
    for (int i = 0; i < n; ++i) {
        const char *nm = moo_algorithm_name(i);
        SET_STRING_ELT(out, i, mkChar(nm ? nm : ""));
    }
    UNPROTECT(1);
    return out;
}

SEXP R_moo_open(SEXP settings)
{
    const char *text = CHAR(STRING_ELT(settings, 0));
    moo_session *s = moo_open(text);
    if (!s) {
        const char *msg = moo_last_error(NULL);
        error("mootation: %s", msg ? msg : "moo_open failed");
    }
    SEXP ext = PROTECT(R_MakeExternalPtr(s, R_NilValue, R_NilValue));
    /* Registered so that a session leaks nothing when R garbage-collects the
       handle without moo_close having been called — which is the normal way an
       R script ends. */
    R_RegisterCFinalizerEx(ext, moo_r_finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}

SEXP R_moo_close(SEXP ext)
{
    moo_session *s = (moo_session *)R_ExternalPtrAddr(ext);
    if (s) {
        moo_close(s);
        R_ClearExternalPtr(ext);
    }
    return R_NilValue;
}

SEXP R_moo_shape(SEXP ext)
{
    moo_session *s = moo_r_handle(ext);
    SEXP out = PROTECT(allocVector(INTSXP, 3));
    INTEGER(out)[0] = moo_n_vars(s);
    INTEGER(out)[1] = moo_n_objs(s);
    INTEGER(out)[2] = moo_n_cons(s);
    UNPROTECT(1);
    return out;
}

SEXP R_moo_generation(SEXP ext)
{
    return ScalarInteger(moo_generation(moo_r_handle(ext)));
}

/* Returns the waiting candidates as a numeric vector of n*n_vars, row-major.
   The R side reshapes with byrow = TRUE. Zero length means the run is over. */
SEXP R_moo_ask(SEXP ext)
{
    moo_session *s = moo_r_handle(ext);
    int n = moo_ask_count(s);
    if (n < 0) error("mootation: %s", moo_last_error(s));
    if (n == 0) return allocVector(REALSXP, 0);

    int nv = moo_n_vars(s);
    SEXP out = PROTECT(allocVector(REALSXP, (R_xlen_t)n * nv));
    if (moo_ask(s, REAL(out), n * nv) < 0) {
        UNPROTECT(1);
        error("mootation: %s", moo_last_error(s));
    }
    UNPROTECT(1);
    return out;
}

/* F and G arrive row-major and flat. Returns the size of the NEXT batch. */
SEXP R_moo_tell(SEXP ext, SEXP F, SEXP G)
{
    moo_session *s = moo_r_handle(ext);
    const double *f = REAL(F);
    const int f_len = (int)LENGTH(F);

    int rc;
    if (moo_n_cons(s) > 0) {
        if (G == R_NilValue || LENGTH(G) == 0)
            error("mootation: this run has %d constraints; pass G",
                  moo_n_cons(s));
        rc = moo_tell(s, f, f_len, REAL(G), (int)LENGTH(G));
    } else {
        rc = moo_tell(s, f, f_len, NULL, 0);
    }
    if (rc < 0) error("mootation: %s", moo_last_error(s));
    return ScalarInteger(rc);
}

/* Returns list(X = ..., F = ..., cv = ...), all flat and row-major. */
SEXP R_moo_result(SEXP ext)
{
    moo_session *s = moo_r_handle(ext);
    int n = moo_result_count(s);
    if (n < 0) error("mootation: %s", moo_last_error(s));
    int nv = moo_n_vars(s), no = moo_n_objs(s);

    SEXP X  = PROTECT(allocVector(REALSXP, (R_xlen_t)n * nv));
    SEXP Fo = PROTECT(allocVector(REALSXP, (R_xlen_t)n * no));
    SEXP cv = PROTECT(allocVector(REALSXP, n));

    if (moo_result(s, REAL(X), n * nv, REAL(Fo), n * no, REAL(cv), n) < 0) {
        UNPROTECT(3);
        error("mootation: %s", moo_last_error(s));
    }

    SEXP out   = PROTECT(allocVector(VECSXP, 3));
    SEXP names = PROTECT(allocVector(STRSXP, 3));
    SET_VECTOR_ELT(out, 0, X);
    SET_VECTOR_ELT(out, 1, Fo);
    SET_VECTOR_ELT(out, 2, cv);
    SET_STRING_ELT(names, 0, mkChar("X"));
    SET_STRING_ELT(names, 1, mkChar("F"));
    SET_STRING_ELT(names, 2, mkChar("cv"));
    setAttrib(out, R_NamesSymbol, names);
    UNPROTECT(5);
    return out;
}
