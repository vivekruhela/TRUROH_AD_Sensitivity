"""Null simulation: does ignoring relatedness inflate false positives for a family-shared ROH, and do the
kinship-aware methods fix it?   python3 tests/validate_kinship.py [replicates] [out.json]

Design (no true ROH effect): sibships of 2-4 with a shared familial liability (SD 1.5 on the logit scale)
and the ROH segment shared within families (as an IBD segment is), plus unrelated singletons.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trurohassoc import firth, glmm  # noqa: E402


def one(rng, n_fam=180, n_single=300, sd=1.5, carrier_family_rate=0.12):
    sizes = rng.integers(2, 5, n_fam)
    fam = np.concatenate([np.repeat(np.arange(n_fam), sizes), n_fam + np.arange(n_single)])
    n = len(fam)
    nf = fam.max() + 1
    u = rng.normal(0, sd, nf)[fam]
    carrier_fam = rng.random(nf) < carrier_family_rate
    roh = (carrier_fam[fam] & (rng.random(n) < 0.7)).astype(float)
    age = rng.normal(0, 1, n)
    y = rng.binomial(1, expit(-0.6 + 0.4 * age + u)).astype(float)
    K = (fam[:, None] == fam[None, :]) * 0.5 + np.eye(n) * 0.5
    X = np.column_stack([np.ones(n), roh - roh.mean(), age])
    out = {}
    full = firth.firth_optimize(X, y)
    out["Firth LRT (ignores kinship)"] = firth.likelihood_test(X, y, full, [1])[1]
    rob, _ = glmm.cluster_robust_cov(X, y, full["beta"], full["covariance"], fam)
    out["Firth + cluster-robust SE"] = float(chi2.sf(full["beta"][1] ** 2 / rob[1, 1], 1))
    null = glmm.fit(X[:, [0, 2]], y, K)
    out["GLMM score (kinship)"] = glmm.score_test(null, X[:, [1]])[1]
    keep = np.unique(fam, return_index=True)[1]          # one random-ish member per family (first listed)
    Xu, yu = X[keep], y[keep]
    if Xu[:, 1].std() > 0:
        fu = firth.firth_optimize(Xu, yu)
        out["Firth, unrelated subset"] = firth.likelihood_test(Xu, yu, fu, [1])[1]
    return out


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(2026)
    t0 = time.time()
    results = {}
    for _ in range(reps):
        for k, v in one(rng).items():
            results.setdefault(k, []).append(v)
    summary = {k: {"replicates": len(v), "type_I_error_at_0.05": float(np.mean(np.array(v) < 0.05)),
                   "MC_SE": float(np.sqrt(0.05 * 0.95 / len(v)))} for k, v in results.items()}
    summary["_seconds"] = round(time.time() - t0, 1)
    print(json.dumps(summary, indent=2))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
