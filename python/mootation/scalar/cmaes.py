# SPDX-License-Identifier: Apache-2.0
"""Single-objective CMA-ES, as N. Hansen, "The CMA Evolution Strategy: A
Tutorial", arXiv:1604.00772 (2016), Figure 6 (Eqs. 38-47) with the default
parameters of its Table 1 (Eqs. 48-58), negative weights (active CMA)
included. Written from the tutorial, not from a library.

The caller asks for λ points, evaluates them and tells their values; the
strategy does the rest. Minimization.
"""

from __future__ import annotations

import math

import numpy as np


class CMAES:
    def __init__(self, mean, sigma: float, rng: np.random.Generator, popsize: int | None = None):
        self.m = np.asarray(mean, float).copy()
        n = self.n = len(self.m)
        self.sigma = float(sigma)
        self.rng = rng
        # Table 1: selection and recombination
        lam = self.lam = int(popsize) if popsize else 4 + int(math.floor(3 * math.log(n)))
        mu = self.mu = lam // 2
        w = np.array([math.log((lam + 1) / 2) - math.log(i) for i in range(1, lam + 1)])  # (49)
        wp, wn = w[:mu], w[mu:]
        self.mueff = wp.sum() ** 2 / (wp ** 2).sum()
        mueff_neg = wn.sum() ** 2 / (wn ** 2).sum() if len(wn) and (wn ** 2).sum() > 0 else 0.0
        # step-size control (55)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.ds = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        # covariance matrix adaptation (56)-(58)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        alpha_cov = 2.0
        self.c1 = alpha_cov / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, alpha_cov * (0.25 + self.mueff + 1 / self.mueff - 2)
                       / ((n + 2) ** 2 + alpha_cov * self.mueff / 2))
        # (50)-(53): positive weights sum to one, negative ones scaled
        alpha_mu = 1 + self.c1 / self.cmu
        alpha_mueff = 1 + 2 * mueff_neg / (self.mueff + 2)
        alpha_posdef = (1 - self.c1 - self.cmu) / (n * self.cmu)
        sum_pos, sum_neg = wp.sum(), -wn.sum()
        self.w = np.where(w >= 0, w / sum_pos,
                          min(alpha_mu, alpha_mueff, alpha_posdef) * w / max(sum_neg, 1e-300))
        self.cm = 1.0                                                              # (54)
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.ps = np.zeros(n)
        self.pc = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.g = 0
        self._y = None

    def ask(self):
        """λ points: x_k = m + σ B D z_k (38)-(40)."""
        z = self.rng.standard_normal((self.lam, self.n))
        self._y = (z * self.D) @ self.B.T
        return self.m + self.sigma * self._y

    def tell(self, values) -> None:
        values = np.asarray(values, float)
        order = np.argsort(values, kind="stable")
        y = self._y[order]
        mu, n = self.mu, self.n
        yw = self.w[:mu] @ y[:mu]                                             # (41)
        self.m = self.m + self.cm * self.sigma * yw                           # (42)
        Cinvsqrt = (self.B / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) \
            * (Cinvsqrt @ yw)                                                 # (43)
        self.g += 1
        norm_ps = float(np.linalg.norm(self.ps))
        hs = norm_ps / math.sqrt(1 - (1 - self.cs) ** (2 * self.g)) \
            < (1.4 + 2 / (n + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + (math.sqrt(self.cc * (2 - self.cc) * self.mueff)
                                             * yw if hs else 0.0)             # (45)
        wo = self.w.copy()                                                    # (46)
        neg = wo < 0
        if neg.any():
            q = np.sum((y[neg] @ Cinvsqrt.T) ** 2, axis=1)
            wo[neg] *= n / np.maximum(q, 1e-300)
        delta_h = (1 - hs) * self.cc * (2 - self.cc)
        self.C = ((1 + self.c1 * delta_h - self.c1 - self.cmu * self.w.sum()) * self.C
                  + self.c1 * np.outer(self.pc, self.pc)
                  + self.cmu * (y.T * wo) @ y)                                # (47)
        self.sigma *= math.exp(self.cs / self.ds * (norm_ps / self.chiN - 1))  # (44)
        self.C = (self.C + self.C.T) / 2
        ev, B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(ev, 1e-300))
        self.B = B

    def spread(self) -> float:
        """σ times the largest standard deviation: the scale of the next step."""
        return self.sigma * float(self.D.max())
