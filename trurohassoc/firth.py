"""Firth-penalised logistic regression with full-design penalised likelihood-ratio tests.

Unchanged numerical core from v1.0.1 (validated against analytic Firth solutions).
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2


class FitFailure(RuntimeError):
    pass


def firth_objective(beta, X, y):
    eta = X @ beta
    prob = expit(eta)
    weight = np.maximum(prob * (1 - prob), 1e-14)
    info = X.T @ (weight[:, None] * X)
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0 or not np.isfinite(logdet):
        raise FitFailure("Non-positive or nonfinite Fisher information determinant")
    inverse = np.linalg.inv(info)
    leverage = weight * np.einsum("ij,jk,ik->i", X, inverse, X)
    likelihood = float(np.sum(y * eta - np.logaddexp(0, eta)) + 0.5 * logdet)
    score = X.T @ (y - prob + leverage * (0.5 - prob))
    return likelihood, score, inverse


def firth_optimize(X, y, *, start=None, fixed=(), max_iter=500, tolerance=1e-5, deadline=None):
    """Constrained optimisation retains the full design's Jeffreys determinant."""
    p = X.shape[1]
    base = np.zeros(p) if start is None else np.asarray(start, float).copy()
    fixed = np.asarray(fixed, dtype=int)
    base[fixed] = 0.0
    fixed_set = set(fixed.tolist())
    free = np.asarray([j for j in range(p) if j not in fixed_set], dtype=int)

    def objective(values):
        if deadline is not None and time.monotonic() > deadline:
            raise FitFailure("Firth fit exceeded the per-model time limit")
        beta = base.copy()
        beta[free] = values
        ll, gradient, _ = firth_objective(beta, X, y)
        return -ll, -gradient[free]

    result = minimize(objective, base[free], jac=True, method="BFGS",
                      options={"maxiter": max_iter, "gtol": tolerance / 10})
    beta = base.copy()
    beta[free] = result.x
    ll, gradient, covariance = firth_objective(beta, X, y)
    residual = float(np.max(np.abs(gradient[free]))) if len(free) else 0.0
    if residual > tolerance or not np.isfinite(beta).all() or not np.isfinite(ll):
        raise FitFailure(f"Firth nonconvergence: max_score={residual:.4g}; {result.message}")
    return dict(beta=beta, covariance=covariance, ll=ll, max_score=residual, iterations=int(result.nit))


def likelihood_test(X, y, full, indices, **kwargs):
    null = firth_optimize(X, y, start=full["beta"], fixed=indices, **kwargs)
    statistic = 2 * (full["ll"] - null["ll"])
    if statistic < -1e-6:
        raise FitFailure("Constrained penalized likelihood exceeds full fit")
    return max(0.0, statistic), float(chi2.sf(max(0.0, statistic), len(indices))), null


def exp_safe(value):
    if not np.isfinite(value):
        return math.nan
    return math.exp(value) if value < 709 else math.inf
