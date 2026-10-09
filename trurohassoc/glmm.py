"""Logistic mixed model with a kinship (GRM) random effect, and cluster-robust variances.

GLMM: AD_i ~ Bernoulli(p_i), logit p = X b + g,  g ~ N(0, tau * K)   (K = 2 x kinship, i.e. GRM scale)
Fitted by penalized quasi-likelihood (PQL) with REML for tau, as in GMMAT::glmmkin. Fixed effects are tested
with the GMMAT/SAIGE-type score test from the null model (tested columns removed); Wald estimates/CIs come
from the full model. Each REML step uses one eigendecomposition of W^1/2 K W^1/2, so every tau evaluation is O(Np).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import chi2

from .firth import FitFailure


def _parts(X, ystar, w, K):
    sw = np.sqrt(w)
    lam, U = np.linalg.eigh(sw[:, None] * K * sw[None, :])
    lam = np.clip(lam, 0.0, None)
    Xt = U.T @ (sw[:, None] * X)
    yt = U.T @ (sw * ystar)
    return dict(lam=lam, U=U, sw=sw, Xt=Xt, yt=yt, logw=float(np.sum(np.log(w))))


def _reml(tau, p):
    d = 1.0 / (1.0 + tau * p["lam"])
    XtD = p["Xt"] * d[:, None]
    A = p["Xt"].T @ XtD
    sign, logdetA = np.linalg.slogdet(A)
    if sign <= 0:
        return np.inf, None
    b = np.linalg.solve(A, XtD.T @ p["yt"])
    r = p["yt"] - p["Xt"] @ b
    quad = float(r @ (d * r))
    logdet_sigma = -p["logw"] + float(np.sum(np.log1p(tau * p["lam"])))
    return 0.5 * (logdet_sigma + logdetA + quad), (b, A, d, r)


def fit(X, y, K, *, tau_start=0.1, tau_max=50.0, max_iter=100, tol=1e-6, deadline=None):
    """Returns dict(beta, covariance, tau, iterations, eta, w, ystar). Raises FitFailure if not converged."""
    import time
    n, k = X.shape
    beta = np.zeros(k)
    beta[0] = np.log(max(y.mean(), 1e-3) / max(1 - y.mean(), 1e-3))
    g = np.zeros(n)
    tau = tau_start
    for iteration in range(1, max_iter + 1):
        if deadline is not None and time.monotonic() > deadline:
            raise FitFailure("GLMM exceeded the per-model time limit")
        eta = X @ beta + g
        mu = np.clip(expit(eta), 1e-10, 1 - 1e-10)
        w = mu * (1 - mu)
        ystar = eta + (y - mu) / w
        p = _parts(X, ystar, w, K)
        res = minimize_scalar(lambda t: _reml(t, p)[0], bounds=(0.0, tau_max), method="bounded",
                              options={"xatol": 1e-6})
        new_tau = float(res.x)
        if _reml(0.0, p)[0] <= _reml(new_tau, p)[0]:
            new_tau = 0.0
        value, (b, A, d, r) = _reml(new_tau, p)
        # BLUP of the random effect: g = tau K Sigma^-1 (y* - X b)
        new_g = new_tau * (K @ (p["sw"] * (p["U"] @ (d * r))))
        change = max(np.max(np.abs(b - beta)), abs(new_tau - tau))
        beta, tau, g = b, new_tau, new_g
        if change < tol and iteration > 1:
            cov = np.linalg.inv(A)
            return dict(beta=beta, covariance=cov, tau=tau, iterations=iteration, parts=p, d=d, r=r,
                        boundary=tau >= tau_max * 0.999)
    raise FitFailure(f"GLMM PQL did not converge in {max_iter} iterations (tau={tau:.3g})")


def score_test(null, G):
    """Score test of extra columns G (n x q, same standardisation as the design) under the fitted null GLMM."""
    p, d = null["parts"], null["d"]
    Gt = p["U"].T @ (p["sw"][:, None] * G)
    XtD = p["Xt"] * d[:, None]
    A = p["Xt"].T @ XtD
    S = Gt.T @ (d * null["r"])                              # G' P y*
    GD = Gt * d[:, None]
    V = Gt.T @ GD - GD.T @ p["Xt"] @ np.linalg.solve(A, XtD.T @ Gt)
    stat = float(S @ np.linalg.solve(V, S))
    return stat, float(chi2.sf(stat, G.shape[1]))


def cluster_robust_cov(X, y, beta, bread, clusters):
    """CR1 sandwich: bread from the (Firth) information, meat from per-cluster summed scores."""
    u = X * (y - expit(X @ beta))[:, None]
    codes, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    sums = np.zeros((len(codes), X.shape[1]))
    np.add.at(sums, inverse, u)
    meat = sums.T @ sums
    G = len(codes)
    factor = G / (G - 1) * (len(y) - 1) / (len(y) - X.shape[1]) if G > 1 else 1.0
    return factor * bread @ meat @ bread, G
