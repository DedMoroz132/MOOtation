# SPDX-License-Identifier: Apache-2.0
"""RWMOP's mechanical, chemical and process problems, RCM01-29 (task 7 of
2026-10-10, §8).

A. Kumar, G. Wu, M. Z. Ali, Q. Luo, R. Mallipeddi, P. N. Suganthan, S. Das,
"A benchmark-suite of real-world constrained multi-objective optimization
problems and some baseline results", Swarm and Evolutionary Computation 67,
100961, 2021, doi:10.1016/j.swevo.2021.100961 (source
rwmop_kumar2021_final-submission, §2.1-2.3, read off the rendered pages; the
long coefficients of RCM21 off the PDF's text). The competition's MATLAB code
has no licence and nothing of it is used: the formulas are the paper's (THE
COMPETITION'S CODE below).

RCM30-50, the power electronics and power system problems, are not here:
the paper does not give what they need — the inverters' switching and
fundamental frequencies and modulation index, with sign patterns s shorter
than the angles Table 2 counts (RCM30-35); the networks' admittance matrices
and loads (RCM36-50). The competition's code has them.

Every function returns (f, g), or (f, g, h) with equalities h = 0 (RCM22-28):
the objectives, and the inequality constraints in the library's sign, g <= 0
satisfied, in the paper's order. evaluate() appends |h| - EQ_EPS to g, as DOC
does (doc.py); the paper's epsilon has no value, EQ_EPS is 1e-4, the CEC
competitions' and minimize's eq_tol.

WHERE A PROBLEM IS NOT DEFINED (task 10 of 2026-10-18, §6). RCM02's f1, a
square root of EI / mu, is undefined where that is negative, which happens
only with the layers out of order (g2 or g4 violated), and RCM24's
equalities, logarithms, outside x8 > 100, x7 < 600, x9 > x7. There the
function returns NaN for every objective and constraint — a failed
evaluation, which minimize's on_failure and a campaign count as such (until
task 10 it raised ValueError, which the core counted and a campaign did not),
and a batch fails in that row alone. UNDEFINED names them;
undefined_share(i) estimates the share of the box where it happens from
100 000 uniform points, and benchmarks.properties carries it.
Integer and discrete variables are rounded inside the function, as the RE
problems do (re.py), and declared in VAR_TYPES for a typed run.

THE PAPER AS PRINTED, except where it cannot be read as it stands:
  missing constants  RCM04's G = 12e6 and tau'' = M R / J (the paper names
          neither; Tanabe and Ishibuchi's RE32 and pymoo's WeldedBeam, the
          same problem, have both); RCM14's rho = 7.8e-6 (kg/mm^3, pymoo's
          Clutch, the same problem; the paper uses rho without a value)
  syntax  RCM08's f1 "4.9x1 6.67x2" is 4.9 x1 + 6.67 x2; RCM07's
          "max{x1, x2.x3, x4}" the maximum of the four
  RCM15   the list of wire diameters reads "0.092, 0.0105, 0.120" and
          "0.307, 0.0331, 0.362", out of its order: 0.105 and 0.331 (RE25's)
  RCM06   g11 "... - 110 <= 0": its left side is at least 754 in the box, so
          no point would be feasible; 1100, as RE35 and the paper's own
          gear box (RCM13, g5) have it
  RCM08   g8 "f1(x) - 4 <= 0": f1 is at least 15.58 in the box, so no point
          would be feasible; f2 - 4, the limit on the door's velocity, as
          RE41 and pymoo's Carside have it
  RCM13   x3 "in {17, 28}": the integers 17 ... 28, as RCM06 writes the same
          variable of the same reducer
  RCM28   h2, h3 "x3 - 0.9 (1 - exp(0.5 x5)) x1": with exp(+) a reactor's
          output can only be <= 0, so h4 (x3 + x4 = 10) could never hold
          (whichever reactor h1 picks); exp(-0.5 x5) and exp(-0.4 x6), the
          conversion 1 - exp(-k t) of a first-order reactor. Then, e.g.,
          x = (11.19, 0, 10, 0, 10, 0, 1, 0) is feasible
  RCM29   g2 "x6^3 + x1^2 + x2^2 + x3^2 = 5.5 <= 0": ... - 5.5 <= 0
Everything else is as printed, the places where the paper differs from the
classic statement of a problem included (they change the problem, not its
readability), listed here for whoever compares with other code:
  RCM08   ten constraints are printed, Table 2 says nine (the ten are used)
  RCM09   no constraint is printed, Table 2 says one (none is used)
  RCM10   P = 104 lb (Koski's two-bar truss has 10^4: with 104 the stress
          constraints never bind)
  RCM11   f3 = 2.62314586e3 x2, f4's exponent -3.975 x2 and g4's 12.098
          (RE61: 305700 * 2289 x2 / (0.06 * 2289)^0.65, -39.75 x2, 2.098)
  RCM14   mu = 0.6, R_sr = 2/3 (r_o^3 - r_i^3) / (r_o^2 r_i^2), V_sr against
          V_sr,max = 10 and T = I_z w / (M_h + M_f) with M_h in N mm
          (pymoo's Clutch: 0.5, (r_o^2 - r_i^2) below, V_sr,max 10 000 mm/s,
          M_h / 1000)
  RCM16   E = 307e8 (steel's is about 207e9)
  RCM17   F_n = 0.5144 / (9.8065 L)^0.5, without V_k, F_c = 4.56e-5 P + 0.2
          and W_s with B^0.6 (RE42: 0.5144 V_k / (g L)^0.5, 0.19 * 24 / 1000
          = 4.56e-3, B^0.7)
  RCM24   f3 = 200 x1 x6 - x5 where h2 has 200 x2 x6 - x5; eight equalities
          printed, Table 2 says six (the eight are used)
  RCM28   Table 2 says four equalities, five are printed (five are used)
THE FRONTS (_rwmop/RCMxx.dat) are this library's, as the task allows for the
references (2026-10-11; build_front, `python -m mootation.benchmarks.rwmop`):
per problem ten seeds of NSGA-II (two objectives, N = 500) or NSGA-III (three
and five, N = 496 and 495), 200 000 evaluations each, ten times the
competition's budget and more, feasibility first; the final populations'
feasible points, the non-dominated ones, at most 2 000 kept by DSS. 2 000
points for every problem but RCM02 (1 658), RCM07 (17: four integers) and
RCM15 (64: an integer and a list). An approximation: front_exact is False,
and IGD+ there is a distance to it; the frame is its corners. Each problem's
construction is its properties' front_built (front_built below).
  RCM22, 23, 25, 26 (reference()): no run meets the equalities to EQ_EPS
(nor do most of the paper's, Tables 6 and 7), so these join the epsilon-
constraint method's points (eps_front: SLSQP for the least f2 under f1 <= c,
every 0/1 setting; two seeds of starts, which agree to 5e-4 in normalised
IGD+ on RCM22 and exactly on the others) with the long runs' feasible points
(none on RCM22 and 23; on RCM25 and 26 no point of either is dominated by
the other's): 472, 524, 2 000 and 687 points.
  RCM28 is exact (rcm28_front, 1 548 points of its two curves; front_exact
True) and RCM29 the best of its sixteen 0/1 settings (rcm29_front): two
points.
  RCM27 is not registered: its front is one point, f = (1.0765, -0.0581) at
x1 = 0.2 + ln 2.1, x3 = 1 (with x3 = 0, g3 holds x1 to 0.2 and gives (1.25,
0.2), dominated), and a one-point reference frames nothing, as RCM24's.
RCM24 (heat exchanger network) is defined but not registered: its
equalities leave one point's worth of front — they force x3 = 0 (h3, h5,
h7), x7 = 100, x8 = 600, x5 = 2e6, x9 = 700, x1 x4 = 0 and x2 x6 = 1e4, and
f2 is h1 itself, 0 there; x1 = 0 then minimises f1 and f3, and f1 = 35
x2^0.6 is least at x2 = 1e4 / 600: the front is the point (35 (50/3)^0.6, 0,
-2e6) up to EQ_EPS, and a one-point reference frames nothing.
RCM19 (multi-product batch plant) is not here: the paper never gives the
demands Q_i its objectives and bounds use.
RCM20 (hydrostatic thrust bearing) is not here: as printed no point is
feasible — with R <= R0 (g6) P0 comes out negative, and the least total
violation of 400 000 points of the box is 1 373 (rcm20 below; the test
rwmop_problems_as_the_paper_prints_them) — while the paper's
Table 5 has its algorithms feasible on it: what ran is not what is printed,
and the source's formulas are not at hand to tell which part is wrong.
The same tables back the two readings above: every algorithm feasible in
every run on RCM06 and RCM08 (Table 3), which as printed have no feasible
point.

THE COMPETITION'S CODE (github.com/P-N-Suganthan/2021-RW-MOP, commit 1e2e7d7,
CEC2021-RWCMOP.zip: CEC2021_func.m, Cal_par.m) was read only to compare
values at points, 200 uniform points of the box per problem (2026-10-11).
Where the code and the paper differ:
  RCM10, RCM12  no difference
  RCM11   f3 = 305700 * 2289 x2 / (0.06 * 2289)^0.65, f4's exponent -39.75 x2
          and g4's 2.098 (the paper: 2.62314586e3 x2, -3.975 x2, 12.098)
  RCM13   f1's terms 14.933 x3 and 10 x3^2 / 3 (the paper: 14.9334 / x3 and
          3.3333 x3^2), f2's constant 1.69e7 (16.91e6), the limits on f2 and
          f3 1300 and 1100 (1100 and 850), and the constraints in the order
          g1-g4, x2 x3 - 40, x1 / x2 - 12, 5 - x1 / x2, the two on x4 and x5,
          then f2's and f3's
  RCM14   the constraints in the order of the paper's g3, g4, g1, g2, g7, g8,
          g5, g6; the values the same
  RCM15   the third variable an index into the list, in [0.51, 42.49] and
          rounded (the paper's is the diameter); the list's 25th value 0.0105,
          as printed in the paper, and its 38th 0.331 (the paper: 0.0331); the
          values otherwise the same
  RCM16   E = 2.07e8 and delta_max = 0.005 (the paper: 307e8 and 0.05)
  RCM17   P with the exponent 2/3 and F_c = 0.19 * 24 P / 1000 + 0.2 (the
          paper: 0.67 and 4.56e-5 P + 0.2)
  RCM19   the demands Q1 = 40 000 and Q2 = 20 000, which the paper does not
          give
  RCM20   P = (log10 log10(8.122e6 mu + 0.8) - 10.04) / -3.55 and dT =
          2 (10^P - 560) (the paper: (... + 3.55) / 10.04 and 559.7); h and W
          with offsets of 1e-5; the constraints W >= 101 000, P0 <= 1000,
          dT <= 50, h >= 0.001, R >= R0, f2 <= 0.001 and W / (pi (R^2 - R0^2))
          <= 5000 (the paper's g1-g7 point the other way, and its g4 bounds P0
          where the code bounds dT). 366 of 400 000 uniform points of the box
          satisfy the code's constraints
  RCM30-50  the code has the data the paper does not give: the modulation
          index, sign patterns and harmonic orders (RCM30-35), the networks'
          conductance and susceptance matrices and loads (RCM36-46), line
          data and loads (RCM47-49) and the cost and emission coefficients
          (RCM50)
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).with_name("_rwmop")

# RCM15's wire diameters (the two out-of-order values read as RE25's)
WIRE = (0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162, 0.0173, 0.018,
        0.020, 0.023, 0.025, 0.028, 0.032, 0.035, 0.041, 0.047, 0.054, 0.063, 0.072, 0.080,
        0.092, 0.105, 0.120, 0.135, 0.148, 0.162, 0.177, 0.192, 0.207, 0.225, 0.244, 0.263,
        0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.500)

SQ2 = math.sqrt(2.0)


def _nearest(values, v):
    return min(values, key=lambda w: abs(w - v))


def rcm01(x):
    """Pressure vessel: x1, x2 integers (thicknesses in 1/16 in), x3, x4."""
    x1, x2 = round(x[0]), round(x[1])
    x3, x4 = x[2], x[3]
    z1, z2 = 0.0625 * x1, 0.0625 * x2
    f1 = 1.7781 * z2 * x3 ** 2 + 0.6224 * z1 * x3 * x4 + 3.1661 * z1 ** 2 * x4 + 19.84 * z1 ** 2 * x3
    f2 = -math.pi * x3 ** 2 * x4 - 4.0 / 3.0 * math.pi * x3 ** 3
    return [f1, f2], [0.00954 * x3 - z2, 0.0193 * x3 - z1]


def rcm02(x):
    """Vibrating platform: d1, d2, d3, b, L."""
    d1, d2, d3, b, L = x
    r1, r2, r3 = 100.0, 2770.0, 7780.0
    E1, E2, E3 = 1.6, 70.0, 200.0
    c1, c2, c3 = 500.0, 1500.0, 800.0
    EI = 2.0 * b / 3.0 * (E1 * d1 ** 3 + E2 * (d2 ** 3 - d1 ** 3) + E3 * (d3 ** 3 - d2 ** 3))
    mu = 2.0 * b * (r1 * d1 + r2 * (d2 - d1) + r3 * (d3 - d2))
    if mu == 0.0 or EI / mu < 0.0:          # only where d1 > d2 or d2 > d3 (g2, g4)
        return [math.nan] * 2, [math.nan] * 5     # undefined: a failed evaluation
    f1 = -math.pi / (2.0 * L ** 2) * math.sqrt(EI / mu)
    f2 = 2.0 * b * L * (c1 * d1 + c2 * (d2 - d1) + c3 * (d3 - d2))
    return [f1, f2], [mu * L - 2800.0, d1 - d2, d2 - d1 - 0.15, d2 - d3, d3 - d2 - 0.01]


def rcm03(x):
    """Two bar truss."""
    x1, x2, x3 = x
    f1 = x1 * math.sqrt(16.0 + x3 ** 2) + x2 * math.sqrt(1.0 + x3 ** 2)
    f2 = 20.0 * math.sqrt(16.0 + x3 ** 2) / (x3 * x1)
    g3 = 80.0 * math.sqrt(1.0 + x3 ** 2) / (x3 * x2) - 1e5
    return [f1, f2], [f1 - 0.1, f2 - 1e5, g3]


def rcm04(x):
    """Welded beam."""
    x1, x2, x3, x4 = x
    P, L, E, G = 6000.0, 14.0, 30e6, 12e6
    tau_max, sigma_max = 13600.0, 30000.0
    f1 = 1.10471 * x1 ** 2 * x2 + 0.04811 * x3 * x4 * (14.0 + x2)
    f2 = 4.0 * P * L ** 3 / (E * x4 * x3 ** 3)
    t1 = P / (SQ2 * x1 * x2)
    M = P * (L + x2 / 2.0)
    R = math.sqrt(x2 ** 2 / 4.0 + ((x1 + x3) / 2.0) ** 2)
    J = 2.0 * (SQ2 * x1 * x2 * (x2 ** 2 / 12.0 + ((x1 + x3) / 2.0) ** 2))
    t2 = M * R / J
    tau = math.sqrt(t1 ** 2 + 2.0 * t1 * t2 * x2 / (2.0 * R) + t2 ** 2)
    sigma = 6.0 * P * L / (x4 * x3 ** 2)
    Pc = 4.013 * E * math.sqrt(x3 ** 2 * x4 ** 6 / 36.0) / L ** 2 \
        * (1.0 - x3 / (2.0 * L) * math.sqrt(E / (4.0 * G)))
    return [f1, f2], [tau - tau_max, sigma - sigma_max, x1 - x4, P - Pc]


def rcm05(x):
    """Disc brake."""
    x1, x2, x3, x4 = x
    s2, s3 = x2 ** 2 - x1 ** 2, x2 ** 3 - x1 ** 3
    f1 = 4.9e-5 * s2 * (x4 - 1.0)
    f2 = 9.82e6 * (s2 / (x3 * x4 * s3))
    return [f1, f2], [20.0 - (x2 - x1), x3 / (3.14 * s2) - 0.4,
                      2.22e-3 * x3 * s3 / s2 ** 2 - 1.0, 900.0 - 2.66e-2 * x3 * x4 * s3 / s2]


def _reducer_stress(a, x2, x3, c, x6):
    return math.sqrt((745.0 * a / (x2 * x3)) ** 2 + c) / (0.1 * x6 ** 3)


def rcm06(x):
    """Speed reducer: x3 an integer."""
    x1, x2, x4, x5, x6, x7 = x[0], x[1], x[3], x[4], x[5], x[6]
    x3 = round(x[2])
    f1 = 0.7854 * x1 * x2 ** 2 * (10.0 * x3 ** 2 / 3.0 + 14.933 * x3 - 43.0934) \
        - 1.508 * x1 * (x6 ** 2 + x7 ** 2) + 7.477 * (x6 ** 3 + x7 ** 3) \
        + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2)
    f2 = _reducer_stress(x4, x2, x3, 1.69e7, x6)
    g = [1.0 / (x1 * x2 ** 2 * x3) - 1.0 / 27.0,
         1.0 / (x1 * x2 ** 2 * x3 ** 2) - 1.0 / 397.5,
         x4 ** 3 / (x2 * x3 * x6 ** 4) - 1.0 / 1.93,
         x5 ** 3 / (x2 * x3 * x7 ** 4) - 1.0 / 1.93,
         x2 * x3 - 40.0,
         x1 / x2 - 12.0,
         -x1 / x2 + 5.0,
         1.9 - x4 + 1.5 * x6,
         1.9 - x5 + 1.1 * x7,
         f2 - 1300.0,
         _reducer_stress(x5, x2, x3, 1.575e8, x7) - 1100.0]       # the paper: 110
    return [f1, f2], g


def rcm07(x):
    """Gear train: four integers, the teeth."""
    x1, x2, x3, x4 = (round(v) for v in x)
    f1 = abs(6.931 - x3 * x4 / (x1 * x2))
    return [f1, float(max(x1, x2, x3, x4))], [f1 / 6.931 - 0.5]


def rcm08(x):
    """Car side impact."""
    x1, x2, x3, x4, x5, x6, x7 = x
    V_mbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
    V_fd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
    f1 = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 1e-5 * x6 + 2.73 * x7
    f2 = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
    f3 = 0.5 * (V_mbp + V_fd)
    g = [-1.0 + 1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3,
         -0.32 + 0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 - 0.019 * x2 * x7 + 0.0144 * x3 * x5
         + 0.0154464 * x6,
         -0.32 + 0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 ** 2,
         -0.32 + 0.214 + 0.00817 * x5 - 0.045195 * x1 - 0.0135168 * x1 + 0.03099 * x2 * x6
         - 0.018 * x2 * x7 + 0.007176 * x3 + 0.023232 * x3 - 0.00364 * x5 * x6 - 0.018 * x2 ** 2,
         -32.0 + 33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728,
         -32.0 + 28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7,
         -32.0 + 46.36 - 9.9 * x2 - 4.4505 * x1,
         f2 - 4.0,                                                # the paper: f1
         V_mbp - 9.9,
         V_fd - 15.7]
    return [f1, f2, f3], g


def rcm09(x):
    """Four bar plane truss: no constraint but the box."""
    x1, x2, x3, x4 = x
    F, E, L = 10.0, 2e5, 200.0
    f1 = L * (2.0 * x1 + SQ2 * x2 + SQ2 * x3 + x4)
    f2 = F * L / E * (2.0 / x1 + 2.0 * SQ2 / x2 - 2.0 * SQ2 / x3 + 2.0 / x4)
    return [f1, f2], []


def rcm10(x):
    """Two bar plane truss."""
    x1, x2 = x
    rho, h, P, E, s0 = 0.283, 100.0, 104.0, 3e7, 2e4
    f1 = 2.0 * rho * h * x2 * math.sqrt(1.0 + x1 ** 2)
    f2 = rho * h * (1.0 + x1 ** 2) ** 1.5 * (1.0 + x1 ** 4) ** 0.5 / (2.0 * SQ2 * E * x1 ** 2 * x2)
    w = (1.0 + x1 ** 2) ** 0.5 / (2.0 * SQ2 * x1 * x2)
    return [f1, f2], [P * (1.0 + x1) * w - s0, P * (-x1 + 1.0) * w - s0]


def rcm11(x):
    """Water resources management: five objectives."""
    x1, x2, x3 = x
    a = x1 * x2
    f = [106780.37 * (x2 + x3) + 61704.67,
         3000.0 * x1,
         2.62314586e3 * x2,
         572250.0 * math.exp(-3.975 * x2 + 9.9 * x3 + 2.74),
         25.0 * (1.39 / a + 4940.0 * x3 - 80.0)]
    g = [-1.0 + (0.00139 / a + 4.94 * x3 - 0.08),
         -1.0 + (0.000306 / a + 1.082 * x3 - 0.0986),
         -50000.0 + (12.307 / a + 49408.24 * x3 + 4051.02),
         -16000.0 + (12.098 / a + 8046.33 * x3 - 696.71),
         -10000.0 + (2.138 / a + 7883.39 * x3 - 705.04),
         -2000.0 + (0.417 * a + 1721.26 * x3 - 136.54),
         -550.0 + (0.164 / a + 631.13 * x3 - 54.48)]
    return f, g


def rcm12(x):
    """Simply supported I-beam."""
    x1, x2, x3, x4 = x
    P, L, E = 600.0, 200.0, 20000.0
    web = x1 - 2.0 * x4
    I4 = x3 * web ** 3 + 2.0 * x2 * x4 * (4.0 * x4 ** 2 + 3.0 * x1 * web)
    f1 = 2.0 * x2 * x4 + x3 * web
    f2 = P * L ** 3 / (4.0 * E * I4)
    g1 = -16.0 + 180000.0 * x1 / I4 + 15000.0 * x2 / (web * x3 ** 3 + 2.0 * x4 * x2 ** 3)
    return [f1, f2], [g1]


def rcm13(x):
    """Gear box: x3 an integer, three objectives."""
    x1, x2, x4, x5, x6, x7 = x[0], x[1], x[3], x[4], x[5], x[6]
    x3 = round(x[2])
    f1 = 0.7854 * x2 ** 2 * x1 * (14.9334 / x3 - 43.0934 + 3.3333 * x3 ** 2) \
        + 0.7854 * (x5 * x7 ** 2 + x4 * x6 ** 2) - 1.508 * x1 * (x7 ** 2 + x6 ** 2) \
        + 7.477 * (x7 ** 3 + x6 ** 3)
    f2 = 10.0 * x6 ** -3 * math.sqrt(16.91e6 + (745.0 * x4 / (x2 * x3)) ** 2)
    f3 = 10.0 * x7 ** -3 * math.sqrt(157.5e6 + (745.0 * x5 / (x2 * x3)) ** 2)
    g = [1.0 / (x1 * x2 ** 2 * x3) - 1.0 / 27.0,
         1.0 / (x1 * x2 ** 2 * x3 ** 2) - 1.0 / 397.5,
         1.0 / (x2 * x6 ** 4 * x3 * x4 ** -3) - 1.0 / 1.93,
         1.0 / (x2 * x7 ** 4 * x3 * x5 ** -3) - 1.0 / 1.93,
         f2 - 1100.0,
         f3 - 850.0,
         x2 * x3 - 40.0,
         -x1 / x2 + 5.0,
         x1 / x2 - 12.0,
         1.5 * x6 - x4 + 1.9,
         1.1 * x7 - x5 + 1.9]
    return [f1, f2, f3], g


def rcm14(x):
    """Multiple disk clutch brake: r_i, r_o, t, F, Z."""
    x1, x2, x3, x4, x5 = x
    rho = 7.8e-6                                  # not given in the paper: pymoo's Clutch
    dR, L_max, mu, V_max, delta, s = 20.0, 30.0, 0.6, 10.0, 0.5, 1.5
    T_max, n, I_z, M_s, M_f, p_max = 15.0, 250.0, 55.0, 40.0, 3.0, 1.0
    M_h = 2.0 / 3.0 * mu * x4 * x5 * (x2 ** 3 - x1 ** 3) / (x2 ** 2 - x1 ** 2)
    w = math.pi * n / 30.0
    A = math.pi * (x2 ** 2 - x1 ** 2)
    p_rz = x4 / A
    R_sr = 2.0 / 3.0 * (x2 ** 3 - x1 ** 3) / (x2 ** 2 * x1 ** 2)
    V_sr = math.pi * R_sr * n / 30.0
    T = I_z * w / (M_h + M_f)
    f1 = math.pi * (x2 ** 2 - x1 ** 2) * x3 * (x5 + 1.0) * rho
    g = [-p_max + p_rz,
         p_rz * V_sr - V_max * p_max,
         dR + x1 - x2,
         -L_max + (x5 + 1.0) * (x3 + delta),
         s * M_s - M_h,
         -T,
         -V_max + V_sr,
         T - T_max]
    return [f1, T], g


def rcm15(x):
    """Spring: x1 an integer (the coils), x2 the coil diameter, x3 the wire's
    from WIRE."""
    x1 = round(x[0])
    x2 = x[1]
    x3 = _nearest(WIRE, x[2])
    C_f = (4.0 * x2 / x3 - 1.0) / (4.0 * x2 / x3 - 4.0) + 0.615 * x3 / x2
    K = 11.5e6 * x3 ** 4 / (8.0 * x1 * x2 ** 3)
    sigma_p = 300.0 / K
    l_f = 1000.0 / K + 1.05 * (x1 + 2.0) * x3
    f1 = math.pi ** 2 * x2 * x3 ** 2 * (x1 + 2.0) / 4.0
    f2 = 8000.0 * C_f * x2 / (math.pi * x3 ** 3)
    g = [f2 - 189000.0, l_f - 14.0, 0.2 - x3, x2 - 3.0, 3.0 - x2 / x3, sigma_p - 6.0,
         sigma_p + 700.0 / K + 1.05 * (x1 + 2.0) * x3 - l_f, 1.25 - 700.0 / K]
    return [f1, f2], g


def rcm16(x):
    """Cantilever beam."""
    x1, x2 = x
    P, E, rho, Sy, d_max = 1.0, 307e8, 7800.0, 3e5, 0.05
    f1 = 0.25 * rho * math.pi * x2 * x1 ** 2
    f2 = 64.0 * P * x2 ** 3 / (3.0 * E * math.pi * x1 ** 4)
    return [f1, f2], [-Sy + 32.0 * P * x2 / (math.pi * x1 ** 3), -d_max + f2]


def rcm17(x):
    """Bulk carrier: L, B, D, T, V_k, C_B; three objectives."""
    L, B, D, T, V_k, C_B = x
    a = 4977.06 * C_B ** 2 - 8105.61 * C_B + 4456.51
    b = -10847.2 * C_B ** 2 + 12817.0 * C_B - 6960.32
    F_n = 0.5144 / (9.8065 * L) ** 0.5
    P = (1.025 * L * B * T * C_B) ** 0.67 * V_k ** 3 / (a + b * F_n)
    W_s = 0.034 * L ** 1.7 * B ** 0.6 * D ** 0.4 * C_B ** 0.5
    W_o = L ** 0.8 * B ** 0.6 * D ** 0.3 * C_B ** 0.1
    W_m = 0.17 * P ** 0.9
    ls = W_s + W_o + W_m
    D_wt = 1.025 * L * B * T * C_B - ls
    F_c = 4.56e-5 * P + 0.2
    D_cwt = D_wt - F_c * (5000.0 * V_k / 24.0 + 5.0) - 2.0 * D_wt ** 0.5
    R_trp = 350.0 / (5000.0 * V_k / 24.0 + 2.0 * (D_cwt / 8000.0 + 0.5))
    ac = D_cwt * R_trp
    S_d = 5000.0 * V_k / 24.0
    C_c = 0.26 * (2000.0 * W_s ** 0.85 + 3500.0 * W_o + 2400.0 * P ** 0.8)
    C_r = 40000.0 * D_wt ** 0.3
    C_v = (105.0 * F_c * S_d + 6.3 * D_wt ** 0.8) * R_trp
    g = [-L / B + 6.0, -15.0 + L / D, -19.0 + L / T, -0.45 * D_wt ** 0.31 + T,
         -0.7 * D - 0.7 + T, -0.32 + F_n,
         -0.53 * T - (0.085 * C_B - 0.002) * B ** 2 / (T * C_B) + (1.0 + 0.52 * D) + 0.07 * B,
         -D_wt + 3000.0, -500000.0 + D_wt]
    return [(C_c + C_r + C_v) / ac, ls, -ac], g


def rcm18(x):
    """Front rail: hh, w, t."""
    hh, w, t = x
    Ea, Fa = 14496.5, 234.9
    E = (-70973.4 + 958.656 * w + 614.173 * hh - 3.827 * w * hh + 57.023 * w * t
         + 63.274 * hh * t - 3.582 * w ** 2 - 1.4842 * hh ** 2 - 1890.174 * t ** 2)
    F = (111.854 - 20.210 * w + 7.560 * hh - 0.025 * w * hh + 2.731 * w * t - 1.479 * hh * t
         + 0.165 * w ** 2)
    g = [-(hh - 136.0) * (146.0 - hh), -(w - 58.0) * (66.0 - w), -(t - 1.4) * (2.2 - t)]
    return [Ea / E, F / Fa], g


# RCM21's response surfaces, coefficients off the PDF's text: the constant,
# x1..x6, x1^2..x6^2, then x_i x_j for i < j in order
_C21 = (
    (1.3667145844797, (-0.00904459793976106, -0.0016193573938033, -0.00758531275221425,
                       -0.00440727360327102, -0.00572216860791644, -0.00936039926190721),
     (2.62510221107328e-6, 4.92982681358861e-7, 2.25524989067108e-6, 1.84605439400301e-6,
      2.17175358243416e-6, 3.90158043948054e-6),
     (4.55276994245781e-7, -6.37013576290982e-7, 8.26736480446359e-7, 5.66352809442276e-8,
      -3.20213897443278e-7, 1.18015467772812e-8, 9.25820391546515e-8, -1.05705364119837e-7,
      -4.74797783014687e-7, -5.02319867013788e-7, 9.54284258085225e-7, 1.80533309229454e-7,
      -1.07938022118477e-6, -1.81370642220182e-7, -2.24238851688047e-7)),
    (-1.19896668942683, (3.04107017009774, 1.23535701600191, 2.13882039381528, 2.33495178382303,
                         2.68632494801975, 3.43918953617606),
     (-7.89144544980703e-4, -2.06085185698215e-4, -7.15269900037858e-4, -7.8449237573837e-4,
      -9.31396896237177e-4, -1.40826531972195e-3),
     (-1.60434988248392e-4, 2.0824655419411e-4, -3.0530659653553e-4, -8.10145973591615e-5,
      6.94728759651311e-5, 1.18015467772812e-8, 9.25820391546515e-8, -1.05705364119837e-7,
      1.69935290196781e-4, 2.32421829190088e-5, -2.0808624041163476e-4, 1.75576341867273e-5,
      2.68422081654044e-4, 4.39852066801981e-5, 2.96785446021357e-5)),
)


def _surface(c, x):
    c0, lin, sq, cross = c
    v = c0 + sum(a * xi for a, xi in zip(lin, x)) + sum(a * xi * xi for a, xi in zip(sq, x))
    k = 0
    for i in range(6):
        for j in range(i + 1, 6):
            v += cross[k] * x[i] * x[j]
            k += 1
    return v


def rcm21(x):
    """Crash energy management for a high-speed train."""
    f1, f2 = _surface(_C21[0], x), _surface(_C21[1], x)
    return [f1, f2], [f1 - 5.0, -f1, f2 - 28.0, -f2]


# ── the chemical and process problems (RCM22-29) ─────────────────────────────
# Each returns (f, g, h): h the equalities, h = 0 wanted; evaluate() appends
# |h| - EQ_EPS to g, as DOC does (doc.py), and equalities() gives h itself.

def rcm22(x):
    """Haverly's pooling problem."""
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
    f = [-9.0 * x1 - 15.0 * x2 + 6.0 * x3 + 16.0 * x4, 10.0 * (x5 + x6)]
    h = [x7 + x8 - x4 - x3, x1 - x5 - x7, x2 - x6 - x8, x9 * x7 + x9 * x8 - 3.0 * x3 - x4]
    g = [x9 * x7 + 2.0 * x5 - 2.5 * x1, x9 * x8 + 2.0 * x6 - 1.5 * x2]
    return f, g, h


def rcm23(x):
    """Reactor network design."""
    x1, x2, x3, x4, x5, x6 = x
    k1 = 0.09755988
    k2 = 0.99 * k1
    k3 = 0.0391908
    k4 = 0.9 * k3
    f = [-x4, x5 ** 0.5 + x6 ** 0.5]
    h = [k1 * x5 * x2 + x1 - 1.0, k3 * x5 * x3 + x3 + x1 - 1.0, k2 * x6 * x2 - x1 + x2,
         k4 * x6 * x4 + x2 - x1 + x4 - x3]
    return f, [x5 ** 0.5 + x6 ** 0.5 - 4.0], h


def rcm24(x):
    """Heat exchanger network design."""
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
    if not (x8 > 100.0 and x7 < 600.0 and x9 > x7):
        return [math.nan] * 3, [], [math.nan] * 8   # the logarithms undefined: a failed evaluation
    f = [35.0 * x1 ** 0.6 + 35.0 * x2 ** 0.6, 200.0 * x1 * x4 - x3, 200.0 * x1 * x6 - x5]
    h = [200.0 * x1 * x4 - x3, 200.0 * x2 * x6 - x5, x3 - 10000.0 * (x7 - 100.0),
         x5 - 10000.0 * (300.0 - x7), x3 - 10000.0 * (600.0 - x8), x5 - 10000.0 * (900.0 - x9),
         x4 * math.log(x8 - 100.0) - x4 * math.log(600.0 - x7) - x8 + x7 + 500.0,
         x6 * math.log(x9 - x7) - x6 * math.log(600.0) - x9 + x7 + 600.0]
    return f, [], h


def rcm25(x):
    """Process synthesis: x2 in {0, 1}."""
    x1, x2 = x[0], round(x[1])
    return [x2 + 2.0 * x1, -x1 ** 2 - x2], [-x1 ** 2 - x2 + 1.25, x1 + x2 - 1.6], []


def rcm26(x):
    """Process synthesis and design: x3 in {0, 1}."""
    x1, x2, x3 = x[0], x[1], round(x[2])
    f = [-x3 + x2 + 2.0 * x1, -x1 ** 2 - x2 + x1 * x3]
    return f, [x2 - x1 + x3], [-2.0 * math.exp(-x2) + x1]


def rcm27(x):
    """Process flow sheeting: x3 in {0, 1}."""
    x1, x2, x3 = x[0], x[1], round(x[2])
    f = [-0.7 * x3 + 0.8 + 5.0 * (0.5 - x1) ** 2, x1 - x3]
    return f, [-math.exp(x1 - 0.2) - x2, x2 + 1.1 * x3 + 1.0, x1 - x3 - 0.2], []


def rcm28(x):
    """Two reactor problem: x7, x8 in {0, 1}."""
    x1, x2, x3, x4, x5, x6 = x[:6]
    x7, x8 = round(x[6]), round(x[7])
    f = [7.5 * x7 + 5.5 * x8 + 7.0 * x5 + 6.0 * x6 + 5.0 * (x1 + x2), x1 + x2]
    h = [x7 + x8 - 1.0,
         x3 - 0.9 * (1.0 - math.exp(-0.5 * x5)) * x1,            # the paper: exp(0.5 x5)
         x4 - 0.8 * (1.0 - math.exp(-0.4 * x6)) * x2,            # the paper: exp(0.4 x6)
         x3 + x4 - 10.0,
         x3 * x7 + x4 * x8 - 10.0]
    g = [x5 - 10.0 * x7, x6 - 10.0 * x8, x1 - 20.0 * x7, x2 - 20.0 * x8]
    return f, g, h


def rcm29(x):
    """Process synthesis: x4 ... x7 in {0, 1}."""
    x1, x2, x3 = x[:3]
    x4, x5, x6, x7 = (round(v) for v in x[3:])
    tail = (1.0 - x1) ** 2 + (2.0 - x2) ** 2 + (3.0 - x3) ** 2
    f = [(1.0 - x4) ** 2 + (1.0 - x5) ** 2 + (1.0 - x6) ** 2 - math.log(1.0 + x7) + tail, tail]
    g = [x1 + x2 + x3 + x4 + x5 + x6 - 5.0,
         x6 ** 3 + x1 ** 2 + x2 ** 2 + x3 ** 2 - 5.5,            # the paper: "= 5.5 <= 0"
         x1 + x4 - 1.2, x2 + x5 - 1.8, x3 + x6 - 2.5, x1 + x7 - 1.2,
         x5 ** 2 + x2 ** 2 - 1.64, x6 ** 2 + x3 ** 2 - 4.25, x5 ** 2 + x3 ** 2 - 4.64]
    return f, g, []


# number -> (function, bounds, objectives)
PROBLEMS = {
    1: (rcm01, [(1, 99), (1, 99), (10, 200), (10, 200)], 2),
    2: (rcm02, [(0.05, 0.5), (0.2, 0.5), (0.2, 0.6), (0.35, 0.5), (3, 6)], 2),
    3: (rcm03, [(1e-5, 100), (1e-5, 100), (1, 3)], 2),
    4: (rcm04, [(0.125, 5), (0.1, 10), (0.1, 10), (0.125, 5)], 2),
    5: (rcm05, [(55, 80), (75, 110), (1000, 3000), (11, 20)], 2),
    6: (rcm06, [(2.6, 3.6), (0.7, 0.8), (17, 28), (7.3, 8.3), (7.3, 8.3), (2.9, 3.9), (5, 5.5)], 2),
    7: (rcm07, [(12, 60)] * 4, 2),
    8: (rcm08, [(0.5, 1.5), (0.45, 1.35), (0.5, 1.5), (0.5, 1.5), (0.875, 2.625), (0.4, 1.2),
                (0.4, 1.2)], 3),
    9: (rcm09, [(1, 3), (SQ2, 3), (SQ2, 3), (1, 3)], 2),
    10: (rcm10, [(0.1, 2), (0.5, 2.5)], 2),
    11: (rcm11, [(0.01, 0.45), (0.01, 0.1), (0.01, 0.1)], 5),
    12: (rcm12, [(10, 80), (10, 50), (0.9, 5), (0.9, 5)], 2),
    13: (rcm13, [(2.6, 3.6), (0.7, 0.8), (17, 28), (7.3, 8.3), (7.3, 8.3), (2.9, 3.9), (5, 5.5)], 3),
    14: (rcm14, [(60, 80), (90, 110), (1, 3), (0, 1000), (2, 9)], 2),
    15: (rcm15, [(1, 70), (0.6, 3), (0.009, 0.5)], 2),
    16: (rcm16, [(0.01, 0.05), (0.2, 1)], 2),
    17: (rcm17, [(150, 274.32), (20, 32.31), (13, 25), (10, 11.71), (14, 18), (0.63, 0.75)], 3),
    18: (rcm18, [(136, 146), (56, 68), (1.4, 2.2)], 2),
    21: (rcm21, [(1.3, 1.7), (2.5, 3.5), (1.3, 1.7), (1.3, 1.7), (1.3, 1.7), (1.3, 1.7)], 2),
    22: (rcm22, [(0, 100), (0, 200), (0, 100), (0, 100), (0, 100), (0, 100), (0, 200), (0, 100),
                 (0, 200)], 2),
    23: (rcm23, [(0, 1)] * 4 + [(1e-5, 16)] * 2, 2),
    24: (rcm24, [(0, 10), (0, 200), (0, 100), (0, 200), (1000, 2000000), (0, 600), (100, 600),
                 (100, 600), (100, 900)], 3),
    25: (rcm25, [(0, 1.6), (0, 1)], 2),
    26: (rcm26, [(0.5, 1.4), (0.5, 1.4), (0, 1)], 2),
    27: (rcm27, [(0.2, 1), (-2.22554, -1), (0, 1)], 2),
    28: (rcm28, [(0, 100)] * 6 + [(0, 1)] * 2, 2),
    29: (rcm29, [(0, 100)] * 3 + [(0, 1)] * 4, 2),
}

# the problems in the registry: those with a reference front (_rwmop/)
REGISTERED = tuple(range(1, 19)) + (21, 22, 23, 25, 26, 28, 29)

# the equalities' count (RCM22-28; evaluate() appends |h| - EQ_EPS to g)
N_EQ = {22: 4, 23: 4, 24: 8, 26: 1, 28: 5}
# the paper's epsilon has no value (its Eq. 2): 1e-4, the CEC competitions'
# and minimize's eq_tol, as DOC's (doc.py)
EQ_EPS = 1e-4

# the integer and discrete variables (mootation.var_types), the rest real
VAR_TYPES = {
    1: ["int", "int", "real", "real"],
    6: ["real", "real", "int", "real", "real", "real", "real"],
    7: ["int"] * 4,
    13: ["real", "real", "int", "real", "real", "real", "real"],
    15: ["int", "real", ("discrete", list(WIRE))],
    25: ["real", "int"],
    26: ["real", "real", "int"],
    27: ["real", "real", "int"],
    28: ["real"] * 6 + ["int"] * 2,
    29: ["real"] * 3 + ["int"] * 4,
}


def name(i: int) -> str:
    return f"RCM{i:02d}"


def evaluate(i: int, x) -> tuple:
    """(objectives, constraints) of RCM<i> at x; an equality h counts as
    |h| - EQ_EPS <= 0, after the inequalities."""
    out = PROBLEMS[i][0]([float(v) for v in x])
    g = list(out[1]) + ([abs(v) - EQ_EPS for v in out[2]] if len(out) == 3 else [])
    return [float(v) for v in out[0]], [float(v) for v in g]


def equalities(i: int, x) -> list:
    """RCM<i>'s equalities h(x) themselves, [] for a problem without."""
    out = PROBLEMS[i][0]([float(v) for v in x])
    return [float(v) for v in out[2]] if len(out) == 3 else []


# The problems that are not defined in part of their box (WHERE A PROBLEM IS NOT
# DEFINED above), and where.
UNDEFINED = {2: "f1 = sqrt(EI / mu) where EI / mu < 0: the layers out of order",
             24: "the logarithms of h7 and h8 outside x8 > 100, x7 < 600, x9 > x7"}


@lru_cache(maxsize=None)
def undefined_share(i: int, n: int = 100_000, seed: int = 0) -> float:
    """The share of n uniform points of RCM<i>'s box where it is not defined
    (its objectives NaN); 0 for a problem defined everywhere (not in UNDEFINED)."""
    if i not in UNDEFINED:
        return 0.0
    box = np.asarray(PROBLEMS[i][1], float)
    X = box[:, 0] + np.random.default_rng(seed).random((n, len(box))) * (box[:, 1] - box[:, 0])
    bad = sum(not math.isfinite(PROBLEMS[i][0](list(map(float, x)))[0][0]) for x in X)
    return bad / n


def n_cons(i: int) -> int:
    lo = [b[0] for b in PROBLEMS[i][1]]
    return len(evaluate(i, lo)[1])


# the competition's budget (the paper, Eq. 117): by M and D
def max_evaluations(i: int) -> int:
    m, d = PROBLEMS[i][2], len(PROBLEMS[i][1])
    return {2: (20000, 80000), 3: (26250, 105000), 4: (35750, 143000),
            5: (53000, 212000)}[m][d > 10]


@lru_cache(maxsize=None)
def _front(key: str) -> np.ndarray:
    return np.loadtxt(DATA / f"{key}.dat", ndmin=2)


def front(key: str) -> np.ndarray:
    """The reference front of RCM<i> (THE FRONTS above), a copy."""
    return _front(key).copy()


def frame(key: str) -> tuple:
    """(ideal, nadir): the reference front's corners."""
    F = _front(key)
    return tuple(float(v) for v in F.min(0)), tuple(float(v) for v in F.max(0))


def build_front(i: int, seeds=range(1, 11), budget: int = 200_000) -> np.ndarray:
    """RCM<i>'s reference front as THE FRONTS above build it."""
    import mootation
    from ..run.archive import dss_order
    from ..run.metrics import nondominated_mask
    _, box, m = PROBLEMS[i]
    alg, pop = ("nsga3", 495) if m == 5 else ("nsga3", 496) if m == 3 else ("nsga2", 500)
    k = n_cons(i)
    neq = N_EQ.get(i, 0)
    X = []
    for seed in seeds:
        kw = dict(algorithm=alg, pop_size=pop, max_evaluations=budget, seed=seed)
        if k - neq:
            kw.update(constraints=lambda x: evaluate(i, x)[1][:k - neq], n_cons=k - neq)
        if neq:          # the equalities' tolerance shrinks from 1 to EQ_EPS over the run
            kw.update(eq_constraints=lambda x: equalities(i, x), n_eq=neq, eq_tol=EQ_EPS,
                      eq_tol_schedule="linear", eq_tol0=1.0)
        r = mootation.minimize(lambda x: evaluate(i, x)[0], box, m, **kw)
        X += [list(x) for x, c in zip(r.variables, r.cv) if c <= 0]
    F = []
    for x in X:
        f, g = evaluate(i, x)
        if all(v <= 0 for v in g) and np.all(np.isfinite(f)):
            F.append(f)
    if not F:
        return np.zeros((0, m))
    F = np.unique(np.array(F), axis=0)
    F = F[nondominated_mask(F)]
    if len(F) > 2000:
        F = F[dss_order(F, 2000, ideal=F.min(0), nadir=F.max(0))]
    return F


# SLSQP ends on an active constraint up to rounding: a point within this of
# feasible counts as feasible in eps_front and rcm29_front (only there)
FEAS_TOL = 1e-9


def eps_front(i: int, n: int = 300, starts: int = 8, seed: int = 1) -> np.ndarray:
    """RCM<i>'s reference front by the epsilon-constraint method, for the small
    problems with equalities or 0/1 variables (THE FRONTS): for every value of
    the 0/1 variables, f2 least subject to the constraints and f1 <= c, by
    SciPy's SLSQP from the previous solution and `starts` random points, c on
    n steps between f1's least value and its value where f2 is least; a point
    kept when evaluate() finds it feasible (|h| <= EQ_EPS) up to FEAS_TOL."""
    import itertools
    from scipy.optimize import minimize as slsqp
    fn, box, m = PROBLEMS[i]
    types = VAR_TYPES.get(i, ["real"] * len(box))
    ints = [j for j, t in enumerate(types) if t == "int"]
    cont = [j for j in range(len(box)) if j not in ints]
    lo = np.array([box[j][0] for j in cont], float)
    hi = np.array([box[j][1] for j in cont], float)
    rng = np.random.default_rng(seed)
    F = []

    def feasible(x):
        try:
            f, g = evaluate(i, x)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        return f if all(v <= FEAS_TOL for v in g) else None

    for combo in itertools.product(*[range(int(box[j][0]), int(box[j][1]) + 1) for j in ints]):
        def full(z, _c=combo):
            x = np.zeros(len(box))
            x[cont] = z
            x[ints] = _c
            return x

        def parts(z):
            try:
                out = fn(list(full(z)))
            except (ValueError, ZeroDivisionError, OverflowError):
                return None
            return (np.array(out[0], float), np.array(out[1], float),
                    np.array(out[2] if len(out) == 3 else [], float))

        # a constraint the 0/1 values alone decide (RCM28's h1) is checked
        # once here, not handed to SLSQP, whose Jacobian it would make singular
        probe = [parts(lo + (hi - lo) * rng.random(len(lo))) for _ in range(3)]
        probe = [q for q in probe if q is not None]
        if not probe:
            continue
        fixed_g = np.all([q[1] == probe[0][1] for q in probe], axis=0)
        fixed_h = np.all([q[2] == probe[0][2] for q in probe], axis=0)
        if np.any(probe[0][1][fixed_g] > 0) or np.any(np.abs(probe[0][2][fixed_h]) > EQ_EPS):
            continue
        live_g, live_h = ~fixed_g, ~fixed_h

        def solve(obj, cap, z0):
            cons = [{"type": "ineq", "fun": lambda z: -parts(z)[1][live_g]}] if live_g.any() else []
            if live_h.any():
                cons.append({"type": "eq", "fun": lambda z: parts(z)[2][live_h]})
            if cap is not None:
                cons.append({"type": "ineq", "fun": lambda z: cap - parts(z)[0][0]})
            r = slsqp(lambda z: parts(z)[0][obj], z0, method="SLSQP",
                      bounds=list(zip(lo, hi)), constraints=cons,
                      options={"maxiter": 300, "ftol": 1e-12})
            return r.x, feasible(full(r.x))

        def best(obj, cap, warm):
            found = []
            for z0 in ([warm] if warm is not None else []) + \
                    [lo + (hi - lo) * rng.random(len(lo)) for _ in range(starts)]:
                if parts(z0) is None:
                    continue
                try:
                    z, f = solve(obj, cap, z0)
                except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                    continue
                if f is not None:
                    found.append((f[obj], f, z))
            return min(found, key=lambda t: t[0]) if found else None

        a, b = best(0, None, None), best(1, None, None)
        if a is None or b is None:
            continue
        F += [a[1], b[1]]
        warm = a[2]
        for c in np.linspace(a[1][0], b[1][0], n):
            got = best(1, c, warm)
            if got is not None:
                F.append(got[1])
                warm = got[2]
    if not F:
        return np.zeros((0, m))
    from ..run.metrics import nondominated_mask
    F = np.unique(np.array(F, float), axis=0)
    return F[nondominated_mask(F)]


def rcm28_front(n: int = 2000) -> np.ndarray:
    """RCM28's front, exact (THE FRONTS): h1 picks one reactor. With x7 = 1,
    g2 and g4 force x6 = x2 = 0, so x4 = 0 (h3), x3 = 10 (h4, h5) and h2 gives
    x1 = 10 / (0.9 (1 - exp(-0.5 x5))), within x1 <= 20 (g3) from x5 = -2
    ln(4/9) to 10 (g1): f = (7.5 + 7 x5 + 5 x1, x1). With x8 = 1, the same
    with x2 = 10 / (0.8 (1 - exp(-0.4 x6))), x6 from -2.5 ln(3/8) to 10:
    f = (5.5 + 6 x6 + 5 x2, x2). The non-dominated points of both curves."""
    from ..run.metrics import nondominated_mask
    x5 = np.linspace(-2.0 * math.log(4.0 / 9.0), 10.0, n)
    x1 = 10.0 / (0.9 * (1.0 - np.exp(-0.5 * x5)))
    x6 = np.linspace(-2.5 * math.log(3.0 / 8.0), 10.0, n)
    x2 = 10.0 / (0.8 * (1.0 - np.exp(-0.4 * x6)))
    F = np.vstack([np.column_stack([7.5 + 7.0 * x5 + 5.0 * x1, x1]),
                   np.column_stack([5.5 + 6.0 * x6 + 5.0 * x2, x2])])
    return F[nondominated_mask(F)]


def rcm29_front() -> np.ndarray:
    """RCM29's front (THE FRONTS): with the 0/1 variables fixed, f1 is f2
    plus a constant and the least f2 is a convex problem (a squared distance
    over linear and convex quadratic constraints), solved by SLSQP from five
    starts in [0, 2]^3; the front is the non-dominated of the sixteen."""
    import itertools
    from scipy.optimize import minimize as slsqp
    from ..run.metrics import nondominated_mask
    F = []
    for c in itertools.product((0, 1), repeat=4):
        def parts(z, _c=c):
            f, g, _ = rcm29(list(z) + list(_c))
            return np.array(f), np.array(g)
        best = None
        for s in range(5):
            z0 = np.random.default_rng(s).random(3) * 2.0
            r = slsqp(lambda z: parts(z)[0][1], z0, method="SLSQP", bounds=[(0.0, 100.0)] * 3,
                      constraints=[{"type": "ineq", "fun": lambda z: -parts(z)[1]}],
                      options={"ftol": 1e-12, "maxiter": 500})
            f, g = evaluate(29, list(r.x) + list(c))
            if all(v <= FEAS_TOL for v in g) and (best is None or f[1] < best[1]):
                best = f
        if best is not None:
            F.append(best)
    F = np.array(F, float)
    return F[nondominated_mask(F)]


def front_built(i: int) -> str | None:
    """How RCM<i>'s reference front was built (THE FRONTS), for the
    properties' front_built; None for RCM28's, which is exact."""
    if i == 28:
        return None
    if i == 29:
        return ("the least f2 of each of the sixteen 0/1 settings, by SLSQP from five starts, "
                "and the non-dominated of them: two points (rwmop.rcm29_front)")
    if i in (22, 23, 25, 26):
        return ("the epsilon-constraint method (SLSQP for the least f2 under f1 <= c, every 0/1 "
                "setting, two seeds of starts) joined with the feasible points of ten long "
                "runs (rwmop.eps_front, rwmop.build_front)")
    return ("ten runs of NSGA-II (two objectives, N = 500) or NSGA-III (three and five), 200 000 "
            "evaluations each, feasibility first: the final populations' feasible "
            "non-dominated points, at most 2 000 kept by DSS (rwmop.build_front)")


def reference(i: int) -> np.ndarray:
    """RCM<i>'s reference front as THE FRONTS build it."""
    if i == 28:
        return rcm28_front()
    if i == 29:
        return rcm29_front()
    if i not in (22, 23, 25, 26):
        return build_front(i)
    from ..run.archive import dss_order
    from ..run.metrics import nondominated_mask
    F = np.vstack([eps_front(i, seed=1, starts=8), eps_front(i, seed=2, starts=16), build_front(i)])
    F = np.unique(F, axis=0)
    F = F[nondominated_mask(F)]
    if len(F) > 2000:
        F = F[dss_order(F, 2000, ideal=F.min(0), nadir=F.max(0))]
    return F


def rcm20(x):
    """Hydrostatic thrust bearing as printed (R, R0, mu, Q): not registered,
    no point is feasible (the module's header); kept for the check."""
    R, R0, mu, Q = x
    P = (math.log10(math.log10(8.122e6 * mu + 0.8)) + 3.55) / 10.04
    dT = 2.0 * (10.0 ** P - 559.7)
    E_f = 9336.0 * Q * 0.0307 * 0.5 * dT
    h = (2.0 * math.pi * 750.0 / 60.0) ** 2 * 2.0 * math.pi * mu / E_f * (R ** 4 / 4.0 - R0 ** 4 / 4.0)
    P0 = 6.0 * mu * Q / (math.pi * h ** 3) * math.log(R / R0)
    W = math.pi * P0 / 2.0 * (R ** 2 - R0 ** 2) / math.log(R / R0)
    f1 = (Q * P0 / 0.7 + E_f) / 12.0
    f2 = 0.0307 / (386.4 * P0) * Q / (2.0 * math.pi * R * h)
    g = [1000.0 - P0, W - 101000.0, 5000.0 - W / (math.pi * (R ** 2 - R0 ** 2)), 50.0 - P0,
         0.001 - 0.0307 / (386.4 * P0) * (Q / (2.0 * math.pi * R * h)), R - R0, h - 0.001]
    return [f1, f2], g


if __name__ == "__main__":
    # python -m mootation.benchmarks.rwmop [OUT]: the reference fronts anew
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA
    out.mkdir(parents=True, exist_ok=True)
    for i in REGISTERED:
        np.savetxt(out / f"{name(i)}.dat", reference(i), fmt="%.17g")
        print(name(i), flush=True)
