# SPDX-License-Identifier: Apache-2.0
#
# MOOtation from R.
#
# R cannot call the C ABI directly: `.C()` marshals only atomic vectors, so an
# opaque session pointer cannot survive it, and `.Call()` needs functions with
# the R API signature. mootation_r.c is the adapter that bridges the two; build
# it first:
#
#     R CMD SHLIB mootation_r.c -L<dir with libmootation> -lmootation \
#         -I<repo>/include
#
# then:
#
#     dyn.load("mootation_r.so")      # or .dll on Windows
#     source("mootation.R")
#
#     zdt1 <- function(x) {
#         g <- 1 + 9 * sum(x[-1]) / (length(x) - 1)
#         c(x[1], g * (1 - sqrt(x[1] / g)))
#     }
#
#     res <- moo_minimize(zdt1, "
#         algorithm = nsga2
#         pop_size  = 40
#         max_gen   = 100
#         n_vars    = 10
#         n_objs    = 2
#         lower     = 0
#         upper     = 1
#     ")
#     plot(res$F, xlab = "f1", ylab = "f2")
#
# NOT RUN ON THE MACHINE THIS WAS WRITTEN ON — no R was installed. The C ABI
# underneath is tested (capi/smoke.c, capi/ctypes_demo.py); this layer is not.
# Verify before relying on it.
#
# R matrices are column-major and the ABI is row-major, so every reshape below
# passes byrow = TRUE or transposes. That is deliberate and confined here.

moo_version <- function() .Call("R_moo_version")

moo_algorithms <- function() .Call("R_moo_algorithms")

moo_open <- function(settings) {
    h <- .Call("R_moo_open", as.character(settings))
    shape <- .Call("R_moo_shape", h)
    structure(list(handle = h,
                   n_vars = shape[1],
                   n_objs = shape[2],
                   n_cons = shape[3]),
              class = "moo_session")
}

moo_close <- function(s) invisible(.Call("R_moo_close", s$handle))

moo_generation <- function(s) .Call("R_moo_generation", s$handle)

# An (n x n_vars) matrix; zero rows when the run has finished.
# The row count is the ALGORITHM's choice — generational algorithms hand over a
# whole offspring generation, steady-state ones one candidate. Never assume
# pop_size.
moo_ask <- function(s) {
    flat <- .Call("R_moo_ask", s$handle)
    if (length(flat) == 0) return(matrix(numeric(0), nrow = 0, ncol = s$n_vars))
    matrix(flat, ncol = s$n_vars, byrow = TRUE)
}

# F is (n x n_objs). Returns the size of the NEXT batch, 0 when finished.
moo_tell <- function(s, F, G = NULL) {
    fflat <- as.numeric(t(as.matrix(F)))
    gflat <- if (is.null(G)) NULL else as.numeric(t(as.matrix(G)))
    .Call("R_moo_tell", s$handle, fflat, gflat)
}

moo_result <- function(s) {
    r <- .Call("R_moo_result", s$handle)
    n <- length(r$cv)
    list(X  = matrix(r$X, nrow = n, ncol = s$n_vars, byrow = TRUE),
         F  = matrix(r$F, nrow = n, ncol = s$n_objs, byrow = TRUE),
         cv = r$cv)
}

moo_minimize <- function(f, settings, constraints = NULL) {
    s <- moo_open(settings)
    on.exit(moo_close(s), add = TRUE)

    X <- moo_ask(s)
    while (nrow(X) > 0) {
        F <- t(apply(X, 1, function(row) as.numeric(f(row))))
        if (is.null(constraints)) {
            moo_tell(s, F)
        } else {
            G <- t(apply(X, 1, function(row) as.numeric(constraints(row))))
            moo_tell(s, F, G)
        }
        X <- moo_ask(s)
    }
    moo_result(s)
}
