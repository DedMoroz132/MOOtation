# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# The least-squares problems of the registry at one objective (task 18,
# stage 1, §1d): MGH and NIST, evaluated by the C++ suite (suite/mgh.hpp,
# suite/nist.hpp; the families "MGH" and "NIST"). The objective is
# 1/2 ||r||^2, the residuals and (MGH) their Jacobian beside it; there is no
# box (bounds -inf / +inf) and every problem has its start x0.
#
# MGH. J.J. Moré, B.S. Garbow, K.E. Hillstrom, "Testing unconstrained
# optimization software", ACM TOMS 7(1) 1981 (source mgh-test-problems_more1981):
# the "Nonlinear Least Squares" collection of §4, its 18 problems at the 28
# sizes (n, m) of its Table IV, each from the starts Table IV runs: x_s, and
# where §5 (c) names NTRIES = 3 also 10 x_s and 100 x_s — the 54 runs of the
# table (MGH<fid>_n<n>_m<m>_x<factor>_1D). f_opt is the minimum (d) of §3 gives
# for that size, halved (MGH's f is the sum of squares): exactly where (d)
# gives it exactly (0, or a closed form; f_opt_kind "exact"), else computed
# (LSQ-1; "computed"). f_local: the other minima (d) prints, halved (LSQ-3).
# large_residual: the global minimum is not zero (the property "large
# residual" of the candidates' list, X9).
#
# NIST. The 27 StRD nonlinear regression files, each from its start 1 and its
# start 2; f_opt the certified residual sum of squares, halved ("certified").
# NIST Standard Reference Data (SRD 140); © U.S. Secretary of Commerce on
# behalf of the United States of America. The files are in the library
# (suite/nist_data.hpp, from tests/data/nist-strd-nls, whose README has the
# source and each file's sha256).
#
# READINGS:
#   LSQ-1. (d) prints twelve of its minima to six digits, truncated
#     ("8.21487 ... 10^-3"). Their f_opt is our lm's from x_s (suite/mgh.hpp's
#     residuals and Jacobian, ftol = xtol = 1e-15; scalar/lm.hpp), to full
#     precision: twice it agrees with (d) to every printed digit (MGH_COMPUTED;
#     python/test_run_layer.py repeats the runs). The minima in closed form
#     (problems 32 to 34) and the zeros are exact.
#   LSQ-2. Chebyquad (35) at n = 1, m = 8 has no minimum in (d), which gives
#     them for m = n only: f_opt None and large_residual None (Table IV's
#     ||f|| is MINPACK's result, not a minimum).
#   LSQ-3. The local minima (d) prints beside the global one — Freudenstein and
#     Roth (2) 48.9842... at (11.41..., -0.8968...), Bard (8) 17.4286... at
#     (0.8406..., -inf, -inf), Kowalik and Osborne (15) 1.02734... 10^-3 at
#     (+inf, -14.07..., -inf, -inf), Brown almost-linear (27) 1 at
#     (0, ..., 0, n + 1) — are levels, halved as printed (six digits, but 1),
#     not targets; f_opt is the global minimum.
#   LSQ-4. The budget is MGH's Table I's limit, 1000 evaluations; NIST names
#     none, and takes the same. pop_size 1: these are problems for the local
#     solvers (lm, lbfgsb), which a box-less problem with a start x0 is for.
#   LSQ-5. NTRIES: §4 says only that a function with rapidly growing
#     subfunctions, such as exponentials, should get NTRIES = 1; §5 (c) lists
#     the problems where it was 3 (4 to 11, 14, 15 with n = 1, 16 with
#     n = 10). We take that list, not a judgement of our own of which
#     functions grow fast.
#   LSQ-6. NIST certifies its sums of squares to 10 or 11 significant digits:
#     a Δf below about 1e-10 of f_opt says nothing.
# ============================================================================
from __future__ import annotations

from fractions import Fraction

# §4's least-squares list (its number -> the function of §3), as
# tests/test_lm_sets.cpp runs Table IV
NLLS = {1: 32, 2: 33, 3: 34, 4: 1, 5: 7, 6: 13, 7: 2, 8: 8, 9: 15, 10: 10, 11: 20, 12: 12, 13: 6,
        14: 16, 15: 35, 16: 27, 17: 17, 18: 19}


def _linear_rank_1(m: int) -> Fraction:              # (33): m (m - 1) / (2 (2m + 1))
    return Fraction(m * (m - 1), 2 * (2 * m + 1))


def _linear_rank_1_zero(m: int) -> Fraction:         # (34): (m^2 + 3m - 6) / (2 (2m - 3))
    return Fraction(m * m + 3 * m - 6, 2 * (2 * m - 3))


# Table IV's sizes (function of §3, n, m), (d)'s minimum there (MGH's f, the
# sum of squares; None: (d) gives none, LSQ-2) and its NTRIES (LSQ-5)
MGH_ROWS = (
    (32, 5, 10, Fraction(10 - 5), 1), (32, 5, 50, Fraction(50 - 5), 1),
    (33, 5, 10, _linear_rank_1(10), 1), (33, 5, 50, _linear_rank_1(50), 1),
    (34, 5, 10, _linear_rank_1_zero(10), 1), (34, 5, 50, _linear_rank_1_zero(50), 1),
    (1, 2, 2, 0, 3), (7, 3, 3, 0, 3), (13, 4, 4, 0, 3), (2, 2, 2, 0, 3),
    (8, 3, 15, "8.21487e-3", 3), (15, 4, 11, "3.07505e-4", 3), (10, 3, 16, "87.9458", 3),
    (20, 6, 31, "2.28767e-3", 3), (20, 9, 31, "1.39976e-6", 3), (20, 12, 31, "4.72238e-10", 3),
    (12, 3, 10, 0, 1), (6, 2, 10, "124.362", 1), (16, 4, 20, "85822.2", 3),
    (35, 1, 8, None, 3), (35, 8, 8, "3.51687e-3", 1), (35, 9, 9, 0, 1), (35, 10, 10, "6.50395e-3", 1),
    (27, 10, 10, 0, 3), (27, 30, 30, 0, 1), (27, 40, 40, 0, 1),
    (17, 5, 33, "5.46489e-5", 1), (19, 11, 65, "4.01377e-2", 1),
)

# LSQ-1: 1/2 ||r||^2 at lm's minimum from x_s, for the minima (d) prints as
# decimals; 2 f agrees with the printed digits
MGH_COMPUTED = {
    (8, 3, 15): 0.0041074386532894795,      # (d) 8.21487e-3
    (15, 4, 11): 0.00015375280192461878,    # (d) 3.07505e-4
    (10, 3, 16): 43.97292758534564,         # (d) 87.9458
    (20, 6, 31): 0.001143835026776221,      # (d) 2.28767e-3
    (20, 9, 31): 6.998800690478596e-07,     # (d) 1.39976e-6
    (20, 12, 31): 2.361190551313011e-10,    # (d) 4.72238e-10
    (6, 2, 10): 62.18109117780744,          # (d) 124.362
    (16, 4, 20): 42911.10081317827,         # (d) 85822.2
    (35, 8, 8): 0.0017584368628389645,      # (d) 3.51687e-3
    (35, 10, 10): 0.003251977400441146,     # (d) 6.50395e-3
    (17, 5, 33): 2.7324473487414097e-05,    # (d) 5.46489e-5
    (19, 11, 65): 0.020068868146773895,     # (d) 4.01377e-2
}

# LSQ-3: the local minima (d) prints, MGH's f, by function of §3
MGH_LOCAL = {2: "48.9842", 8: "17.4286", 15: "1.02734e-3", 27: 1}

# suite/nist.hpp's kNames
NIST_NAMES = ("Misra1a", "Chwirut2", "Chwirut1", "Lanczos3", "Gauss1", "Gauss2", "DanWood", "Misra1b",
              "Kirby2", "Hahn1", "Nelson", "MGH17", "Lanczos1", "Lanczos2", "Gauss3", "Misra1c",
              "Misra1d", "Roszman1", "ENSO", "MGH09", "Thurber", "BoxBOD", "Rat42", "MGH10", "Eckerle4",
              "Rat43", "Bennett5")

BUDGET = 1000                                        # LSQ-4


def factors(tries: int) -> tuple:
    """The starts' factors of x_s: x_s, 10 x_s, 100 x_s (§4, NTRIES)."""
    return (1, 10, 100)[:tries]


def mgh_name(fid: int, n: int, m: int, factor: int) -> str:
    return f"MGH{fid:02d}_n{n:02d}_m{m:02d}_x{factor}_1D"


def nist_name(name: str, start: int) -> str:
    return f"NIST_{name}_s{start}_1D"


def half(v) -> float:
    """An exact minimum of (d), a printed level or a certified sum of squares, halved: 1/2 ||r||^2."""
    return float(Fraction(v) / 2)


def mgh_f_opt(fid: int, n: int, m: int, d) -> tuple:
    """(f_opt, f_opt_kind) of a row of MGH_ROWS: (None, None) where (d) gives none."""
    if d is None:
        return None, None
    if isinstance(d, str):
        return MGH_COMPUTED[(fid, n, m)], "computed"
    return half(d), "exact"


def mgh_local(fid: int) -> tuple:
    """The local levels of function fid, halved (LSQ-3)."""
    return (half(MGH_LOCAL[fid]),) if fid in MGH_LOCAL else ()


def mgh_spec(fid: int, n: int, m: int) -> dict:
    return {"family": "MGH", "n_var": n, "n_obj": 1, "mgh_id": fid, "mgh_m": m}


def nist_spec(name: str, n: int) -> dict:
    return {"family": "NIST", "n_var": n, "n_obj": 1, "nist": name}
