"""Publication figures (Nature Methods conventions).

* widths 89 mm (single) / 183 mm (double); sans-serif 5-7 pt; 0.6 pt axes; no chart titles
* bold lowercase panel letters; colour-blind-safe Okabe-Ito hues, identity always also labelled
* every figure saved as vector PDF + SVG (editable text) and 600-dpi PNG + LZW TIFF
* odds ratios on log axes with a reference line at 1; open markers / dashed whiskers = estimate
  outside the observed support (extrapolation), never hidden
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

from .common import ANCESTRY_NAMES  # noqa: E402

MM = 1 / 25.4
SINGLE, DOUBLE = 89 * MM, 183 * MM
ANC_COLOR = {"EUR": "#0072B2", "NAM": "#E69F00", "AFR": "#009E73", "EAS": "#CC79A7", "SAS": "#56B4E9"}
ROH_COLOR, NONROH_COLOR, INK, MUTED, GRID = "#D55E00", "#5B5B5B", "#1A1A1A", "#6B6B6B", "#D9D9D9"

STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 6.5, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "lines.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "legend.frameon": False, "savefig.dpi": 600, "figure.dpi": 150, "axes.unicode_minus": False,
}


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 1e-3:
        e = int(math.floor(math.log10(p)))
        return f"{p / 10 ** e:.1f}×10$^{{{e}}}$"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def fmt_p_plain(p):
    if p is None or not np.isfinite(p):
        return "NA"
    return f"{p:.1e}" if p < 1e-3 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}")


def panel(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def empty(ax, message):
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=6, wrap=True)


def log_or_axis(ax, lo, hi, axis="x"):
    lo, hi = max(lo, 1e-3), min(hi, 1e3)
    span = [v for v in [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 10, 25, 50, 100] if lo * 0.8 <= v <= hi * 1.25]
    if len(span) > 7:
        span = [v for v in span if v in (0.01, 0.1, 0.25, 0.5, 1, 2, 4, 10, 100)]
    if axis == "x":
        ax.set_xscale("log"); ax.set_xlim(lo / 1.15, hi * 1.15)
        ax.xaxis.set_major_locator(FixedLocator(span)); ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{v:g}" for v in span]); ax.axvline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=0)
    else:
        ax.set_yscale("log"); ax.set_ylim(lo / 1.15, hi * 1.15)
        ax.yaxis.set_major_locator(FixedLocator(span)); ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticklabels([f"{v:g}" for v in span]); ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=0)


def bounds(*frames, cols=("CI_LOW", "CI_HIGH"), clip=(0.02, 50)):
    vals = []
    for f in frames:
        if f is not None and len(f):
            for c in cols:
                v = pd.to_numeric(f[c], errors="coerce").to_numpy(float)
                vals.extend(v[np.isfinite(v) & (v > 0)])
    if not vals:
        return 0.25, 4
    return max(min(min(vals), 0.8), clip[0]), min(max(max(vals), 1.25), clip[1])


def save(fig, out_dir, stem):
    out_dir = Path(out_dir)
    paths = {}
    for ext in ("pdf", "svg", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=600 if ext == "png" else None)
        paths[ext] = path
    try:
        path = out_dir / f"{stem}.tiff"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        paths["tiff"] = path
    except Exception:  # noqa: BLE001 - Pillow without TIFF support
        pass
    plt.close(fig)
    return paths


def _one(results, model, test):
    s = results[(results.MODEL == model) & (results.TEST == test)]
    return s.iloc[0] if len(s) else None


def _coef(coefs, model, term):
    if coefs is None or coefs.empty:
        return None
    s = coefs[(coefs.MODEL == model) & (coefs.TERM == term)]
    return s.iloc[0] if len(s) else None


def _draw_ci(ax, x, y, lo, hi, color, horizontal=True, filled=True, size=3.4, marker="o", lw=0.9):
    ls = "-" if filled else (0, (2, 1.5))
    if horizontal:
        ax.plot([lo, hi], [y, y], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=2)
    else:
        ax.plot([x, x], [lo, hi], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=2)
    ax.plot(x, y, marker=marker, ms=size, mfc=color if filled else "white", mec=color, mew=0.8, zorder=3, ls="none")


# ----------------------------------------------------------------------------- Figure 1
FOREST_ROWS = [
    ("Original TRUROH run", [("orig:M0", "M0: ROH only (original)"), ("orig:M1", "M1: + age, sex, PC1–5 (original)")]),
    ("Covariates and APOE", [("A0_unadjusted", "ROH only"), ("A1_cov_original", "+ age, sex, PC1–5"),
                              ("A1m_cov_APOEsample", "+ age, sex, PC1–5 (APOE-typed samples)"),
                              ("P1_ROH_cov_APOE", "Model 1: + APOE ε4 dosage"),
                              ("A2_carrier", "+ APOE ε4 carrier"), ("A3_genotypic", "+ APOE ε4 genotype"),
                              ("A6_stratum_carrier0", "ε4 non-carriers only"), ("A6_stratum_carrier1", "ε4 carriers only")]),
    ("Global ancestry", [("G0_Q_instead_of_PCs", "global ancestry instead of PCs")] +
     [(f"G1_{a}_adjust", f"Model 1 + global {a}") for a in ("NAM", "EUR", "AFR")]),
    ("Local ancestry at ROH locus", [(f"L1_{a}_local", f"Model 1 + local + global {a}") for a in ("NAM", "EUR", "AFR")] +
     [(f"L1n_{a}_local_noglobal", f"Model 1 + local {a} (no global)") for a in ("NAM", "EUR", "AFR")] +
     [("L4_joint_local", "Model 1 + joint local + global ancestry")]),
    ("Relatedness (kinship)", [("K1_P1_GLMM", "Model 1, GLMM with kinship random effect"),
                               ("K3_P1_robust", "Model 1, family-cluster-robust SE"),
                               ("K5_P1_unrelated", "Model 1, unrelated subset"),
                               ("K7_P1_FROH", "Model 1 + genome-wide F_ROH"),
                               ("K8_P1_FROH_GLMM", "Model 1 + F_ROH, GLMM with kinship")]),
]


def fig_roh_forest(results, region_id, out_dir, cohort):
    r = results[results.REGION_ID == region_id]
    items = []
    for section, rows in FOREST_ROWS:
        block = []
        for key, label in rows:
            if key.startswith("orig:"):
                row = r[(r.SOURCE == "original_run") & (r.MODEL == key[5:])]
            else:
                row = r[(r.MODEL == key) & (r.TEST == "ROH")]
            if len(row):
                block.append((label, row.iloc[0], key == "P1_ROH_cov_APOE"))
        if block:
            items.append((section, block))
    nrows = sum(len(b) + 1 for _, b in items)
    fig = plt.figure(figsize=(DOUBLE, 0.36 + nrows * 0.138))
    ax = fig.add_axes([0.37, 0.09, 0.28, 0.86])
    est = [row for _, b in items for _, row, _ in b]
    lo, hi = bounds(pd.DataFrame(est))
    y, ticks = 0, []
    for section, block in items:
        ax.text(-1.23, y, section, transform=ax.get_yaxis_transform(), fontweight="bold", va="center", fontsize=6.5)
        y += 1
        for label, row, primary in block:
            converged = row.get("CONVERGED") == 1 and np.isfinite(row.get("OR", np.nan))
            ax.text(-1.20, y, label, transform=ax.get_yaxis_transform(), va="center", fontsize=6,
                    fontweight="bold" if primary else "normal")
            if converged and np.isfinite(row.get("CI_LOW", np.nan)):
                _draw_ci(ax, row.OR, y, max(row.CI_LOW, lo), min(row.CI_HIGH, hi), INK if primary else NONROH_COLOR,
                         marker="D" if primary else "o", size=3.6 if primary else 3.0)
                txt = f"{row.OR:.2f} ({row.CI_LOW:.2f}–{row.CI_HIGH:.2f})"
            elif converged:
                ax.plot(row.OR, y, "o", ms=3, mfc="white", mec=NONROH_COLOR)
                txt = f"{row.OR:.2f} (exact)"
            else:
                txt = str(row.get("STATUS", "not estimated")).replace("_", " ")
            ncar = row.get("CARRIERS")
            ax.text(1.04, y, f"{int(row.CARRIER_CASES) if np.isfinite(row.get('CARRIER_CASES', np.nan)) else '–'}/"
                             f"{int(row.CARRIER_CONTROLS) if np.isfinite(row.get('CARRIER_CONTROLS', np.nan)) else '–'}",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=6)
            ax.text(1.27, y, txt, transform=ax.get_yaxis_transform(), va="center", fontsize=6)
            ax.text(1.78, y, fmt_p(row.get("P_VALUE", np.nan)), transform=ax.get_yaxis_transform(), va="center", fontsize=6)
            ticks.append(y)
            y += 1
    for xpos, head in [(1.04, "Carriers\ncase/ctrl"), (1.27, "OR (95% CI)"), (1.78, "P")]:
        ax.text(xpos, -1.2, head, transform=ax.get_yaxis_transform(), va="bottom", fontsize=6, fontweight="bold")
    ax.set_ylim(y - 0.4, -0.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    log_or_axis(ax, lo, hi)
    ax.set_xlabel(f"Odds ratio of AD for ROH carriers ({region_id})")
    return save(fig, out_dir, f"Fig1_ROH_main_effect_{_safe(region_id)}")


# ----------------------------------------------------------------------------- Figure 2
def fig_apoe(results, coefs, contrasts, preds, region_id, out_dir):
    r = results[results.REGION_ID == region_id]
    c = coefs[coefs.REGION_ID == region_id] if not coefs.empty else coefs
    k = contrasts[contrasts.REGION_ID == region_id] if not contrasts.empty else contrasts
    p = preds[preds.REGION_ID == region_id] if not preds.empty else preds
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.15), gridspec_kw={"width_ratios": [1.05, 1, 1.1], "wspace": 0.55})
    # a: APOE main effects
    ax = axes[0]
    spec = [("P1_ROH_cov_APOE", "APOE4", "per ε4 allele"), ("A2_carrier", "APOE_CARRIER", "ε4 carrier vs none"),
            ("A3_genotypic", "APOE_1", "one ε4 vs none"), ("A3_genotypic", "APOE_2", "two ε4 vs none")]
    rows = [(lab, _coef(c, m, t)) for m, t, lab in spec]
    rows = [(lab, row) for lab, row in rows if row is not None]
    if rows:
        lo, hi = bounds(pd.DataFrame([x for _, x in rows]))
        for i, (lab, row) in enumerate(rows):
            _draw_ci(ax, row.OR, i, row.CI_LOW, row.CI_HIGH, INK)
        ax.set_yticks(range(len(rows)), [lab for lab, _ in rows])
        ax.set_ylim(len(rows) - 0.4, -0.6)
        log_or_axis(ax, lo, hi)
        ax.set_xlabel("OR of AD (APOE ε4)")
        pr = _one(r, "P1_ROH_cov_APOE", "APOE4")
        if pr is not None:
            ax.text(0.98, 0.02, f"N = {int(pr.N):,}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color=MUTED)
    else:
        empty(ax, "APOE not available\n(see apoe_audit.json)")
    panel(ax, "a", x=-0.62)
    # b: ROH effect by APOE
    ax = axes[1]
    kk = k[k.GROUP == "ROH_by_APOE4"] if len(k) else k
    strata = [(_one(r, f"A6_stratum_carrier{v}", "ROH"), v) for v in (0, 1)]
    if len(kk):
        lo, hi = bounds(kk, pd.DataFrame([s for s, _ in strata if s is not None and np.isfinite(s.get("CI_LOW", np.nan))]))
        for _, row in kk.iterrows():
            ok = bool(row.WITHIN_OBSERVED_RANGE)
            _draw_ci(ax, row.VALUE, row.OR, max(row.CI_LOW, lo), min(row.CI_HIGH, hi), ROH_COLOR, horizontal=False, filled=ok)
            ax.text(row.VALUE, max(row.CI_LOW, lo) / 1.12, f"n={int(row.N_AT_VALUE)}", fontsize=5.3, color=MUTED,
                    va="top", ha="center")
        for s, v in strata:
            if s is not None and s.get("CONVERGED") == 1 and np.isfinite(s.get("CI_LOW", np.nan)):
                _draw_ci(ax, 2.85 + 0.6 * v, s.OR, max(s.CI_LOW, lo), min(s.CI_HIGH, hi), NONROH_COLOR,
                         horizontal=False, marker="s", size=3.0)
        ax.set_xticks([0, 1, 2, 2.85, 3.45], ["0", "1", "2", "ε4−", "ε4+"])
        ax.axvline(2.45, color=GRID, lw=0.6)
        ax.text(0.5, 1.0, "interaction model", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.5, color=MUTED)
        ax.text(3.15, 1.0, "ε4 strata", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.5, color=MUTED)
        ax.set_xlim(-0.4, 3.75)
        log_or_axis(ax, lo / 1.25, hi * 2.2, axis="y")
        ax.set_xlabel("APOE ε4 alleles")
        ax.set_ylabel("OR of AD for ROH carriers")
        t = _one(r, "P2_ROHxAPOE", "ROH:APOE4")
        if t is not None:
            ax.text(0.03, 0.91, f"ROH×APOE P = {fmt_p(t.P_VALUE)}", transform=ax.transAxes, fontsize=5.8, va="top")
    else:
        status = _one(r, "P2_ROHxAPOE", "ROH:APOE4")
        empty(ax, "ROH×APOE not estimable\n" + (str(status.STATUS).replace("_", " ") if status is not None else ""))
    panel(ax, "b", x=-0.36)
    # c: predicted risk
    ax = axes[2]
    if len(p) and "APOE4" in p:
        for roh, color, dx, lab in [(0, NONROH_COLOR, -0.09, "no ROH"), (1, ROH_COLOR, 0.09, "ROH carrier")]:
            s = p[p.ROH == roh].sort_values("APOE4")
            ax.plot(s.APOE4 + dx, s.RISK, color=color, lw=0.8, zorder=1)
            for _, row in s.iterrows():
                filled = row.OBSERVED_N >= 5
                _draw_ci(ax, row.APOE4 + dx, row.RISK, row.RISK_LOW, row.RISK_HIGH, color, horizontal=False, filled=filled)
            ax.text(0.04, 0.96 - 0.08 * roh, lab, transform=ax.transAxes, color=color, fontsize=6, va="top",
                    fontweight="bold")
        ax.set_xticks([0, 1, 2])
        ax.set_xlim(-0.35, 2.35)
        ax.set_ylim(0, min(1, p.RISK_HIGH.max() * 1.12))
        ax.set_xlabel("APOE ε4 alleles")
        ax.set_ylabel("Standardized P(AD)")
    else:
        empty(ax, "Predicted risk unavailable")
    panel(ax, "c", x=-0.30)
    return save(fig, out_dir, f"Fig2_APOE_and_ROHxAPOE_{_safe(region_id)}")


# ----------------------------------------------------------------------------- Figure 3
def _curve(ax, k, group_prefix, ancestries, xscale=1.0, label_end=True, pvals=None):
    pvals = pvals or {}
    drew = False
    for a in ancestries:
        s = k[k.GROUP == f"{group_prefix}_{a}"].sort_values("VALUE") if len(k) else k
        if not len(s):
            continue
        drew = True
        color = ANC_COLOR.get(a, MUTED)
        x = s.VALUE.to_numpy(float) * xscale
        inside = s.WITHIN_OBSERVED_RANGE.astype(bool).to_numpy()
        if inside.any():
            ax.fill_between(x[inside], s.CI_LOW[inside], s.CI_HIGH[inside], color=color, alpha=0.15, lw=0, zorder=0)
            ax.plot(x[inside], s.OR[inside], color=color, lw=1.0, zorder=2)
        if label_end:
            j = ancestries.index(a)
            tail = f"  P={fmt_p(pvals[a])}" if a in pvals else ""
            ax.plot([0.03, 0.09], [0.955 - 0.075 * j] * 2, transform=ax.transAxes, color=color, lw=1.2)
            ax.text(0.11, 0.955 - 0.075 * j, a + tail, transform=ax.transAxes, color=color, fontsize=5.6,
                    va="center", fontweight="bold")
    return drew


def _pv(r, model, test, ancestries):
    out = {}
    for a in ancestries:
        row = _one(r, model.format(a=a), test.format(a=a))
        if row is not None and np.isfinite(row.get("P_VALUE", np.nan)):
            out[a] = row.P_VALUE
    return out


def fig_ancestry(results, coefs, contrasts, region_id, ancestries, out_dir, apoe_locus, origin_n=None):
    origin_n = origin_n or {}
    r = results[results.REGION_ID == region_id]
    k = contrasts[contrasts.REGION_ID == region_id] if not contrasts.empty else contrasts
    c = coefs[coefs.REGION_ID == region_id] if not coefs.empty else coefs
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 4.3), gridspec_kw={"hspace": 0.62, "wspace": 0.5})
    # a: ancestry main effects
    ax = axes[0, 0]
    rows = []
    for a in ancestries:
        for model, term, kind in [(f"G1_{a}_adjust", f"Q10_{a}", "global (per 10%)"),
                                  (f"L1_{a}_local", f"LAD_{a}", "local | global"),
                                  (f"L1n_{a}_local_noglobal", f"LAD_{a}", "local, no global")]:
            row = _coef(c, model, term)
            if row is not None:
                rows.append((a, kind, row))
    if rows:
        lo, hi = bounds(pd.DataFrame([x for *_, x in rows]))
        markers = {"global (per 10%)": "s", "local | global": "o", "local, no global": "o"}
        labels = []
        for i, (a, kind, row) in enumerate(rows):
            _draw_ci(ax, row.OR, i, row.CI_LOW, row.CI_HIGH, ANC_COLOR.get(a, MUTED), marker=markers[kind],
                     filled=kind != "local, no global")
            labels.append(f"{a} {kind}")
        ax.set_yticks(range(len(rows)), labels, fontsize=5.5)
        for t, (a, *_x) in zip(ax.get_yticklabels(), rows):
            t.set_color(INK)
        ax.set_ylim(len(rows) - 0.4, -0.6)
        log_or_axis(ax, lo, hi)
        ax.set_xlabel("OR of AD (ancestry)")
    else:
        empty(ax, "Ancestry data unavailable")
    panel(ax, "a", x=-0.75)
    # b: ROH effect by local ancestry dosage
    ax = axes[0, 1]
    if len(k) and _curve(ax, k, "ROH_by_LA", ancestries, pvals=_pv(r, "L2_{a}_ROHxLA", "ROH:LAD_{a}", ancestries)):
        lo, hi = bounds(k[k.GROUP.str.startswith("ROH_by_LA") & k.WITHIN_OBSERVED_RANGE.eq(1)])
        log_or_axis(ax, lo, hi * 3, axis="y")
        ax.set_xlim(-0.05, 2.35); ax.set_xticks([0, 1, 2])
        ax.set_xlabel("Local ancestry at ROH locus (haplotypes)")
        ax.set_ylabel("OR of AD for ROH carriers")
        ax.set_xlim(-0.05, 2.1)
    else:
        empty(ax, "ROH×local ancestry not estimable")
    panel(ax, "b", x=-0.32)
    # c: ROH effect by ancestral origin of the homozygous haplotype
    ax = axes[0, 2]
    orig = k[(k.GROUP == "ROH_by_origin")] if len(k) else k
    pts = []
    for a in ancestries:
        s = orig[(orig.MODEL == f"L5_{a}_origin") & (orig.MODIFIER == a)] if len(orig) else orig
        if len(s):
            pts.append((a, s.iloc[0], _one(r, f"L5_{a}_origin", f"ROH_O_{a}")))
    if pts:
        lo, hi = bounds(pd.DataFrame([x for _, x, _ in pts]))
        for i, (a, row, test) in enumerate(pts):
            _draw_ci(ax, i, row.OR, max(row.CI_LOW, lo), min(row.CI_HIGH, hi), ANC_COLOR.get(a, MUTED), horizontal=False)
            if test is not None:
                ax.text(i, hi * 1.05, f"P={fmt_p(test.P_VALUE)}", ha="center", fontsize=5.2, color=MUTED)
        ax.set_xticks(range(len(pts)), [f"{a}\nn={origin_n.get(a, '?')}" for a, *_ in pts])
        ax.set_xlim(-0.6, len(pts) - 0.4)
        log_or_axis(ax, lo, hi * 1.25, axis="y")
        ax.set_xlabel("Ancestral origin of ROH haplotype")
        ax.set_ylabel("OR of AD for ROH carriers")
    else:
        empty(ax, "Too few carriers per haplotype origin\n(see tables/carrier_diagnostics.tsv)")
    panel(ax, "c", x=-0.32)
    # d: APOE e4 effect by global ancestry
    ax = axes[1, 0]
    if len(k) and _curve(ax, k, "APOE4_by_Q", ancestries, xscale=10, pvals=_pv(r, "G2_{a}_APOExQ", "APOE4:Q10_{a}", ancestries)):
        lo, hi = bounds(k[k.GROUP.str.startswith("APOE4_by_Q") & k.WITHIN_OBSERVED_RANGE.eq(1)])
        log_or_axis(ax, lo, hi * 3, axis="y")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Global ancestry (%)")
        ax.set_ylabel("OR of AD per APOE ε4 allele")
    else:
        empty(ax, "APOE×global ancestry not estimable")
    panel(ax, "d", x=-0.32)
    # e: APOE e4 effect by local ancestry at APOE
    ax = axes[1, 1]
    e = k[k.GROUP.str.startswith("APOE4_by_LAapoe")] if len(k) else k
    if apoe_locus and len(e):
        lo, hi = bounds(e)
        for j, a in enumerate(ancestries):
            s = e[e.GROUP == f"APOE4_by_LAapoe_{a}"]
            for _, row in s.iterrows():
                _draw_ci(ax, row.VALUE + (j - 1) * 0.18, row.OR, max(row.CI_LOW, lo), min(row.CI_HIGH, hi),
                         ANC_COLOR.get(a, MUTED), horizontal=False, filled=bool(row.WITHIN_OBSERVED_RANGE))
            if len(s):
                t = _one(r, f"E1_{a}_APOExLAapoe", f"APOE4:LAA_{a}")
                ax.text(0.03, 0.97 - 0.075 * j, a + (f"  P={fmt_p(t.P_VALUE)}" if t is not None else ""),
                        transform=ax.transAxes, color=ANC_COLOR.get(a, MUTED), fontsize=5.6, fontweight="bold", va="top")
        log_or_axis(ax, lo, hi * 1.6, axis="y")
        ax.set_xticks([0, 1, 2]); ax.set_xlim(-0.5, 2.5)
        ax.set_xlabel("Local ancestry at APOE (haplotypes)")
        ax.set_ylabel("OR of AD per APOE ε4 allele")
    else:
        empty(ax, "Local ancestry at APOE not supplied\n(--apoe-locus-la; optional)")
    panel(ax, "e", x=-0.32)
    # f: ROH effect by global ancestry
    ax = axes[1, 2]
    if len(k) and _curve(ax, k, "ROH_by_Q", ancestries, xscale=10, pvals=_pv(r, "G3_{a}_ROHxQ", "ROH:Q10_{a}", ancestries)):
        lo, hi = bounds(k[k.GROUP.str.startswith("ROH_by_Q") & k.WITHIN_OBSERVED_RANGE.eq(1)])
        log_or_axis(ax, lo, hi * 3, axis="y")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Global ancestry (%)")
        ax.set_ylabel("OR of AD for ROH carriers")
    else:
        empty(ax, "ROH×global ancestry not estimable")
    panel(ax, "f", x=-0.32)
    return save(fig, out_dir, f"Fig3_ancestry_effects_{_safe(region_id)}")


# ----------------------------------------------------------------------------- Figure 4 (supplementary data)
def fig_data(frame, region_id, ancestries, out_dir):
    d = frame[frame.AD.notna()]
    fig = plt.figure(figsize=(DOUBLE, 3.9))
    gs = fig.add_gridspec(2, len(ancestries) + 1, hspace=0.65, wspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
    present = [a for a in ancestries if "LAD_" + a in d]
    if present:
        groups = [("no ROH", d[d.ROH.eq(0)]), ("ROH", d[d.ROH.eq(1)])]
        width, x0 = 0.36, np.arange(len(present))
        shades = {0: 0.25, 1: 0.6, 2: 1.0}
        for gi, (gname, g) in enumerate(groups):
            bottom = np.zeros(len(present))
            for dose in (0, 1, 2):
                frac = np.array([np.isclose(g["LAD_" + a].round(), dose).mean() if len(g) else 0 for a in present])
                cols = [ANC_COLOR.get(a, MUTED) for a in present]
                ax.bar(x0 + (gi - 0.5) * (width + 0.04), frac, width, bottom=bottom, color=cols, alpha=shades[dose],
                       edgecolor="white", linewidth=0.5)
                bottom += frac
        ax.set_xticks(x0, present)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of individuals")
        ax.text(0.5, 1.02, "left: no ROH · right: ROH\nshade: 0 / 1 / 2 haplotypes", transform=ax.transAxes,
                ha="center", fontsize=5.2, color=MUTED, va="bottom")
    else:
        empty(ax, "No local ancestry")
    panel(ax, "a", x=-0.45)
    for j, a in enumerate(present):
        ax = fig.add_subplot(gs[0, j + 1])
        if "Q_" + a in d:
            non, car = d[d.ROH.eq(0)], d[d.ROH.eq(1)]
            jitter = np.random.default_rng(1).normal(0, 0.05, len(non))
            ax.scatter(non["Q_" + a] * 100, non["LAD_" + a] + jitter, s=1.2, color=NONROH_COLOR, alpha=0.25, lw=0, rasterized=True)
            ax.scatter(car["Q_" + a] * 100, car["LAD_" + a], s=7, color=ROH_COLOR, lw=0.3, edgecolor="white", zorder=3)
            rr = np.corrcoef(d["Q_" + a], d["LAD_" + a])[0, 1]
            ax.text(0.03, 0.97, f"r = {rr:.2f}", transform=ax.transAxes, fontsize=5.6, va="top")
            ax.set_xlabel(f"Global {a} (%)")
            ax.set_ylabel(f"Local {a} (haplotypes)")
            ax.set_yticks([0, 1, 2]); ax.set_xlim(0, 100)
            if j == 0:
                ax.text(0.97, 0.03, "ROH carriers", transform=ax.transAxes, ha="right", fontsize=5.5, color=ROH_COLOR)
        else:
            empty(ax, "No global ancestry")
        if j == 0:
            panel(ax, "b", x=-0.4)
    ax = fig.add_subplot(gs[1, 0:2])
    if d.APOE4.notna().any():
        tab = d.dropna(subset=["APOE4"]).groupby(["ROH", "APOE4"]).AD.agg(["mean", "size", "sum"]).reset_index()
        for roh, color, dx, lab in [(0, NONROH_COLOR, -0.18, "no ROH"), (1, ROH_COLOR, 0.18, "ROH")]:
            s = tab[tab.ROH == roh]
            ax.bar(s.APOE4 + dx, s["mean"], 0.34, color=color, label=lab, edgecolor="white", lw=0.5)
            for _, row in s.iterrows():
                ax.text(row.APOE4 + dx, row["mean"] + 0.015, f"{int(row['sum'])}/{int(row['size'])}", ha="center",
                        fontsize=5, color=INK, rotation=0)
        ax.set_xticks([0, 1, 2]); ax.set_ylim(0, 1)
        ax.set_xlabel("APOE ε4 alleles"); ax.set_ylabel("Observed AD fraction (cases/N)")
        ax.legend(loc="upper left", handlelength=1, ncol=2)
    else:
        empty(ax, "APOE not available")
    panel(ax, "c", x=-0.24)
    ax = fig.add_subplot(gs[1, 2:])
    if "ROH_ORIGIN" in d:
        car = d[d.ROH.eq(1)]
        cats = [a for a in ancestries if (car.ROH_ORIGIN == a).any()]
        for i, a in enumerate(cats):
            s = car[car.ROH_ORIGIN == a]
            n_case, n_ctrl = int(s.AD.eq(1).sum()), int(s.AD.eq(0).sum())
            ax.barh(i, n_ctrl, color=ANC_COLOR.get(a, MUTED), alpha=0.45, edgecolor="white", lw=0.5)
            ax.barh(i, n_case, left=n_ctrl, color=ANC_COLOR.get(a, MUTED), edgecolor="white", lw=0.5)
            ax.text(n_ctrl + n_case + 0.4, i, f"{n_case} cases / {n_ctrl} controls", va="center", fontsize=5.5)
        ax.set_yticks(range(len(cats)), cats)
        ax.invert_yaxis()
        ax.set_xlabel("ROH carriers (light: controls, dark: cases)")
        ax.set_xlim(0, max(1, car.ROH_ORIGIN.value_counts().max()) * 1.6)
    else:
        empty(ax, "No local ancestry")
    panel(ax, "d", x=-0.12)
    return save(fig, out_dir, f"FigS1_data_overview_{_safe(region_id)}")


# ----------------------------------------------------------------------------- Figure S2: all tests
FAMILY_LABEL = {"primary": "Primary", "apoe": "APOE", "global_ancestry": "Global\nancestry",
                "local_ancestry": "Local\nancestry", "apoe_locus_ancestry": "APOE-locus\nancestry", "relatedness": "Relatedness", "original_run": "Original\nrun"}


def fig_all_tests(results, out_dir):
    fams = [f for f in FAMILY_LABEL if f in set(results.FAMILY)]
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.6))
    ok = results[results.CONVERGED.eq(1) & results.P_VALUE.notna()]
    rng = np.random.default_rng(3)
    labelled, tick_labels = [], []
    for i, fam in enumerate(fams):
        s = ok[ok.FAMILY == fam]
        x = i + rng.uniform(-0.28, 0.28, len(s))
        y = -np.log10(s.P_VALUE.clip(lower=1e-300))
        ax.scatter(x, y, s=8, color=INK if fam == "primary" else NONROH_COLOR, alpha=0.85, lw=0, zorder=3)
        for xi, yi, (_, row) in zip(x, y, s.iterrows()):
            if row.P_VALUE < 0.05:
                labelled.append((yi, xi, i, f"{row.MODEL.split('_')[0]} {row.TEST}"))
        n_skip = int((results.FAMILY == fam).sum() - len(s))
        tick_labels.append(FAMILY_LABEL[fam].replace("\n", " ") + f"\n{len(s)} fitted" + (f", {n_skip} n.e." if n_skip else ""))
    nominal = -np.log10(0.05)
    ax.axhline(nominal, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.text(len(fams) - 0.45, -np.log10(0.05), "P = 0.05", fontsize=5.5, color=MUTED, va="bottom", ha="right")
    top_y = max([nominal] + [y for y, *_ in labelled])
    placed = {}
    for yi, xi, col, lab in sorted(labelled, reverse=True)[:10]:
        ty = yi
        for other in placed.get(col, []):
            if abs(other - ty) < top_y * 0.06:
                ty = other - top_y * 0.06
        placed.setdefault(col, []).append(ty)
        ax.annotate(lab, (xi, yi), xytext=(col + 0.32, ty), textcoords="data", fontsize=4.8, va="center", color=INK,
                    arrowprops=dict(arrowstyle="-", lw=0.3, color=MUTED, shrinkA=0, shrinkB=1.5))
    ax.set_xticks(range(len(fams)), tick_labels)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(fams) - 0.4)
    top = max(nominal, float((-np.log10(ok.P_VALUE.clip(lower=1e-300))).max()) if len(ok) else 1)
    ax.set_ylim(0, top * 1.15)
    ax.set_ylabel("−log$_{10}$ P")
    return save(fig, out_dir, "FigS2_all_tests_overview")


def _safe(text):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text))


def make_figures(out_dir, results, coefs, contrasts, preds, frames, ancestries, apoe_locus, cohort):
    plt.rcParams.update(STYLE)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    figs = []
    for region_id, frame in frames.items():
        origin_n = frame.loc[frame.ROH.eq(1) & frame.AD.notna(), "ROH_ORIGIN"].value_counts().to_dict() if "ROH_ORIGIN" in frame else {}
        figs.append(("FigS1", region_id, fig_data(frame, region_id, ancestries, out_dir)))
        figs.append(("Fig1", region_id, fig_roh_forest(results, region_id, out_dir, cohort)))
        figs.append(("Fig2", region_id, fig_apoe(results, coefs, contrasts, preds, region_id, out_dir)))
        figs.append(("Fig3", region_id, fig_ancestry(results, coefs, contrasts, region_id, ancestries, out_dir, apoe_locus, origin_n)))
    figs.append(("FigS2", None, fig_all_tests(results, out_dir)))
    order = {"Fig1": 0, "Fig2": 1, "Fig3": 2, "FigS1": 3, "FigS2": 4}
    return sorted(figs, key=lambda x: (order[x[0]], str(x[1])))
