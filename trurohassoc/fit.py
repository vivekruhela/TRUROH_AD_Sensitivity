"""Fit one catalogued model: support checks, Firth fit, LR tests, natural-unit coefficients,
delta-method effect contrasts (e.g. ROH OR at APOE4 = 0/1/2) and g-computed predicted risks."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import time

import numpy as np
from scipy.special import expit
from scipy.stats import fisher_exact, norm

from .catalog import unit_of
from .common import unique
from .firth import FitFailure, exp_safe, firth_optimize, likelihood_test
from . import glmm
from scipy.stats import chi2

Z = 1.959963984540054


def formula(model):
    return "AD ~ 1 + " + " + ".join(model.terms)


def term_values(d, term):
    values = np.ones(len(d))
    for col in term.split(":"):
        values = values * d[col].to_numpy(float)
    return values


def build_design(d, terms, transform=None):
    """Every non-intercept column is centred/scaled for numerical stability; estimates are
    back-transformed exactly (Firth/Jeffreys is equivariant under affine column changes)."""
    arrays, kept, dropped, tf = [np.ones(len(d))], ["INTERCEPT"], [], {}
    for term in terms:
        values = term_values(d, term)
        if transform is not None:
            if term not in transform:
                continue
            m, s = transform[term]
        else:
            if not np.isfinite(values).all():
                raise ValueError(f"nonfinite:{term}")
            s = float(values.std())
            if s < 1e-12:
                dropped.append(term)
                continue
            m = float(values.mean())
        arrays.append((values - m) / s)
        kept.append(term)
        tf[term] = (m, s)
    return np.column_stack(arrays), kept, tf, dropped


def natural(full, kept, tf):
    scale = np.array([1.0] + [tf[t][1] for t in kept[1:]])
    beta = full["beta"] / scale
    cov = full["covariance"] / np.outer(scale, scale)
    return beta, cov


def fit_model(data, model, core, region, args, rng=None, rel=None):
    rng = rng or np.random.default_rng(20261008)
    metadata = {"REGION_ID": region.REGION_ID, "CHR": int(region.CHR), "CONS_START": int(region.CONS_START),
                "CONS_END": int(region.CONS_END), "MODEL": model.name, "LABEL": model.label, "FAMILY": model.family,
                "SOURCE": "this_analysis", "FORMULA": formula(model), "SAMPLE_RULE": model.sample,
                "SUBSET": json.dumps(model.subset or {}, sort_keys=True), "STATUS": "pending", "DETAIL": "",
                "CONVERGED": 0, "METHOD": model.method,
                "P_METHOD": {"fisher": "two-sided Fisher exact", "glmm": "GLMM score test (PQL/REML, kinship random effect)",
                             "firth_robust": "Firth estimate, cluster-robust (CR1) Wald"}.get(model.method, "Firth penalized likelihood ratio (chi-square)"),
                "CI_METHOD": {"fisher": "not calculated", "glmm": "GLMM Wald 95%",
                              "firth_robust": "cluster-robust Wald 95%"}.get(model.method, "Firth-Wald 95%")}
    rows = [dict(metadata, TEST=test, TESTED_TERMS=";".join(terms), DF=len(terms)) for test, terms in model.tests.items()]
    test_family = {test: model.test_family.get(test, model.family) for test in model.tests}
    empty = ([], [], [])

    def finish(status, detail=""):
        for row in rows:
            row.update(metadata, STATUS=status, DETAIL=detail, FAMILY=test_family[row["TEST"]])
        return (rows,) + empty

    raw_terms = unique([x for term in model.terms for x in term.split(":")])
    extra = list(core) if model.sample != "available" else []
    if model.sample == "apoe_dosage_complete":
        extra.append("APOE4")
    if model.sample == "unrelated":
        extra.append("UNRELATED")
    required = unique(["AD", "ROH"] + raw_terms + extra + list((model.subset or {}).keys()))
    missing = [x for x in required if x not in data]
    if missing:
        return finish("unavailable_variables", ";".join(missing))
    if model.method == "glmm" and (rel is None or rel.grm is None):
        return finish("unavailable_kinship", "Supply --kinship (KING table or GCTA GRM) for GLMM models")
    d = data.copy()
    for key, value in (model.subset or {}).items():
        d = d[d[key].eq(value)]
    d = d[np.isfinite(d[required].to_numpy(float)).all(axis=1)].copy()
    if model.sample == "unrelated":
        d = d[d.UNRELATED.eq(1)].copy()
    metadata.update(N=len(d), CASES=int(d.AD.eq(1).sum()), CONTROLS=int(d.AD.eq(0).sum()),
                    CARRIERS=int(d.ROH.eq(1).sum()), CARRIER_CASES=int((d.ROH.eq(1) & d.AD.eq(1)).sum()),
                    CARRIER_CONTROLS=int((d.ROH.eq(1) & d.AD.eq(0)).sum()), NONCARRIERS=int(d.ROH.eq(0).sum()),
                    SAMPLE_SHA256=hashlib.sha256("\n".join(sorted(d.IID)).encode()).hexdigest())
    if not len(d):
        return finish("no_complete_samples", "Required variables missing for everyone (APOE/ancestry?); see audits")
    if d.AD.nunique() != 2:
        return finish("single_outcome_class")
    if min(metadata["CARRIERS"], metadata["NONCARRIERS"]) < args.min_carriers:
        return finish("insufficient_carrier_support", f"Minimum per exposure group: {args.min_carriers}")
    for col in model.discrete_support or []:
        levels = [0, 1, 2] if col == "APOE4" else [0, 1]
        for value, roh in itertools.product(levels, (0, 1)):
            count = int((d[col].eq(value) & d.ROH.eq(roh)).sum())
            if count < args.min_interaction_cell:
                return finish("insufficient_interaction_support",
                              f"{col}={value}, ROH={roh}: N={count}; requires {args.min_interaction_cell}")
    for col in model.carrier_split or []:
        carriers = d[d.ROH.eq(1)]
        for value in (0, 1):
            count = int(carriers[col].eq(value).sum())
            if count < args.min_interaction_cell:
                return finish("insufficient_interaction_support",
                              f"{col}={value} among ROH carriers: N={count}; requires {args.min_interaction_cell}")
    for col in model.varying_support or []:
        for roh in (0, 1):
            values = d.loc[d.ROH.eq(roh), col].to_numpy(float)
            if len(values) < args.min_interaction_cell or np.ptp(values) < 1e-8:
                return finish("insufficient_modifier_variation", f"{col} has inadequate variation/support in ROH={roh}")
    if model.method == "fisher":
        a, b = metadata["CARRIER_CASES"], metadata["CARRIER_CONTROLS"]
        c, dd = metadata["CASES"] - a, metadata["CONTROLS"] - b
        odds, pvalue = fisher_exact([[a, b], [c, dd]], alternative="two-sided")
        for row in rows:
            row.update(metadata, STATUS="converged", CONVERGED=1, OR=float(odds), P_VALUE=float(pvalue), FAMILY=test_family[row["TEST"]])
        return (rows,) + empty
    tested = {term for terms in model.tests.values() for term in terms}
    try:
        X, kept, tf, dropped = build_design(d, model.terms)
    except ValueError as exc:
        return finish("nonfinite_design", str(exc))
    lost = sorted(tested & set(dropped))
    if lost:
        return finish("no_tested_term_variation", ";".join(lost))
    metadata.update(N_PARAMETERS=X.shape[1], DROPPED_CONSTANTS=";".join(dropped), DESIGN_TERMS=";".join(kept))
    if len(d) <= X.shape[1] + 5:
        return finish("insufficient_samples_for_design")
    if np.linalg.matrix_rank(X) != X.shape[1]:
        return finish("rank_deficient", "Contrast not identifiable; tested terms are never silently dropped")
    metadata["DESIGN_CONDITION_NUMBER"] = float(np.linalg.cond(X))
    if metadata["DESIGN_CONDITION_NUMBER"] > 1e8:
        return finish("ill_conditioned_design", f"Condition number {metadata['DESIGN_CONDITION_NUMBER']:.3g} > 1e8")
    options = dict(max_iter=args.max_iterations, tolerance=args.tolerance,
                   deadline=time.monotonic() + args.max_fit_seconds)
    y = d.AD.to_numpy(float)
    K = None
    try:
        if model.method == "glmm":
            K = rel.matrix_for(list(d.IID))
            full = glmm.fit(X, y, K, deadline=options["deadline"], max_iter=args.max_iterations)
        else:
            full = firth_optimize(X, y, **options)
    except (FitFailure, np.linalg.LinAlgError, ValueError, OverflowError, KeyError) as exc:
        return finish("fit_failed", str(exc))
    metadata.update(ITERATIONS=full["iterations"], MAX_SCORE=full.get("max_score"), PENALIZED_LL=full.get("ll"))
    if model.method == "glmm":
        metadata.update(GLMM_TAU=full["tau"], GLMM_TAU_AT_BOUND=int(full["boundary"]))
    if model.method == "firth_robust":
        if "CLUSTER" not in d:
            return finish("unavailable_variables", "CLUSTER")
        robust, n_clusters = glmm.cluster_robust_cov(X, y, full["beta"], full["covariance"], d.CLUSTER.to_numpy())
        carrier_clusters = int(d.loc[d.ROH.eq(1), "CLUSTER"].nunique())
        metadata.update(N_CLUSTERS=n_clusters, CARRIER_CLUSTERS=carrier_clusters)
        full = dict(full, covariance=robust)
    beta, cov = natural(full, kept, tf)
    coefficients = []
    for j, term in enumerate(kept):
        if j == 0:
            continue
        se = math.sqrt(max(float(cov[j, j]), 0.0))
        coefficients.append(dict(REGION_ID=region.REGION_ID, MODEL=model.name, LABEL=model.label, FAMILY=model.family,
                                 TERM=term, UNIT=unit_of(term), BETA=float(beta[j]), SE=se, OR=exp_safe(beta[j]),
                                 CI_LOW=exp_safe(beta[j] - Z * se), CI_HIGH=exp_safe(beta[j] + Z * se),
                                 WALD_P=float(2 * norm.sf(abs(beta[j] / se))) if se > 0 else np.nan,
                                 N=len(d), CARRIERS=metadata["CARRIERS"],
                                 NOTE=("conditional on interacting modifiers = 0" if any(term in t.split(":") and ":" in t for t in kept) else "")))
    by_term = {c["TERM"]: c for c in coefficients}
    for row, terms in zip(rows, model.tests.values()):
        row.update(metadata, FAMILY=test_family[row["TEST"]])
        try:
            indices = [kept.index(term) for term in terms]
            if model.method == "firth":
                statistic, pvalue, null = likelihood_test(X, y, full, indices, **options)
                row.update(STATUS="converged", CONVERGED=1, P_VALUE=pvalue, LR=statistic, NULL_MAX_SCORE=null["max_score"])
            elif model.method == "firth_robust":
                b = full["beta"][indices]
                statistic = float(b @ np.linalg.solve(full["covariance"][np.ix_(indices, indices)], b))
                row.update(STATUS="converged", CONVERGED=1, P_VALUE=float(chi2.sf(statistic, len(indices))), WALD_CHI2=statistic)
            else:
                rest = [j for j in range(X.shape[1]) if j not in indices]
                null = glmm.fit(X[:, rest], y, K, tau_start=full["tau"], deadline=options["deadline"], max_iter=args.max_iterations)
                statistic, pvalue = glmm.score_test(null, X[:, indices])
                row.update(STATUS="converged", CONVERGED=1, P_VALUE=pvalue, SCORE_CHI2=statistic, GLMM_TAU_NULL=null["tau"])
            if len(terms) == 1:
                row.update({key: by_term[terms[0]][key] for key in ["BETA", "SE", "OR", "CI_LOW", "CI_HIGH"]})
                row["UNIT"] = by_term[terms[0]]["UNIT"]
        except (FitFailure, np.linalg.LinAlgError, ValueError, OverflowError) as exc:
            row.update(STATUS="restricted_fit_failed", CONVERGED=0, DETAIL=str(exc))

    contrasts = []
    for spec in model.contrasts:
        if not set(spec["weights"]).issubset(kept):
            continue
        w = np.zeros(len(kept))
        for term, weight in spec["weights"].items():
            w[kept.index(term)] = weight
        est = float(w @ beta)
        se = math.sqrt(max(float(w @ cov @ w), 0.0))
        mod = spec["modifier"]
        observed = d[mod].to_numpy(float) if mod in d else None
        carriers_mod = d.loc[d.ROH.eq(1), mod].to_numpy(float) if mod in d else None
        base = [t for t in spec["weights"] if ":" not in t and not t.startswith("ROH_O_")]
        reference_set = carriers_mod if base and base[0] == "ROH" else observed
        inside = (reference_set is None or not len(reference_set) or
                  (np.quantile(reference_set, 0.025) - 1e-9 <= spec["value"] <= np.quantile(reference_set, 0.975) + 1e-9))
        contrasts.append(dict(REGION_ID=region.REGION_ID, MODEL=model.name, GROUP=spec["group"], EFFECT_OF=base[0] if base else "ROH",
                              MODIFIER=mod, VALUE=spec["value"], BETA=est, SE=se, OR=exp_safe(est),
                              CI_LOW=exp_safe(est - Z * se), CI_HIGH=exp_safe(est + Z * se),
                              WALD_P=float(2 * norm.sf(abs(est / se))) if se > 0 else np.nan,
                              WITHIN_OBSERVED_RANGE=int(bool(inside)),
                              N_AT_VALUE=int(np.sum(np.isclose(reference_set, spec["value"]))) if reference_set is not None else np.nan))
    predictions = []
    if model.predict:
        draws = rng.multivariate_normal(full["beta"], full["covariance"], size=args.prediction_draws, method="svd")
        names = list(model.predict)
        for combo in itertools.product(*[model.predict[n] for n in names]):
            frame = d.copy()
            for n, v in zip(names, combo):
                frame[n] = float(v)
            Xp, keptp, _, _ = build_design(frame, model.terms, transform=tf)
            if keptp != kept:
                continue
            point = float(np.mean(expit(Xp @ full["beta"])))
            sims = expit(Xp @ draws.T).mean(axis=0)
            obs = d.copy()
            for n, v in zip(names, combo):
                obs = obs[obs[n].eq(v)]
            predictions.append(dict(REGION_ID=region.REGION_ID, MODEL=model.name, **{n: v for n, v in zip(names, combo)},
                                    RISK=point, RISK_LOW=float(np.quantile(sims, .025)), RISK_HIGH=float(np.quantile(sims, .975)),
                                    OBSERVED_N=len(obs), OBSERVED_CASE_FRACTION=float(obs.AD.mean()) if len(obs) else np.nan))
    return rows, coefficients, contrasts, predictions


ADJUSTED_COLUMNS = ["Q_VALUE", "BONFERRONI_P", "FDR_ASSOCIATED"]


def original_test_rows(original, hc_ids):
    import pandas as pd
    rows = []
    if original.empty or "REGION_ID" not in original:
        return rows
    # P values are nominal: multiple-testing columns of older TRUROH runs (< 2.2.0) are not carried over
    original = original.drop(columns=[c for c in ADJUSTED_COLUMNS if c in original])
    for record in original.loc[original.REGION_ID.isin(hc_ids)].to_dict("records"):
        row = dict(record, SOURCE="original_run", FAMILY="original_run", TEST=record.get("TESTED_TERM", "original_test"),
                   LABEL="original TRUROH model " + str(record.get("MODEL", "")),
                   TESTED_TERMS=record.get("TESTED_TERM", ""), METHOD="original_Firth",
                   FORMULA="See original association_models.json", SAMPLE_RULE="original_run", DF=1)
        for key in ["N", "CASES", "CONTROLS", "CARRIERS", "CARRIER_CASES", "CARRIER_CONTROLS", "NONCARRIERS",
                    "CONVERGED", "OR", "CI_LOW", "CI_HIGH", "BETA", "SE", "P_VALUE"]:
            row[key] = pd.to_numeric(record.get(key), errors="coerce")
        if row.get("CONVERGED") != 1:
            row["P_VALUE"] = np.nan
        rows.append(row)
    return rows
