"""Self-contained HTML report + summary.json + interpretation.md."""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd

from . import VERSION
from .catalog import FAMILIES
from .common import ANCESTRY_NAMES, write_json
from .figures import fmt_p_plain, make_figures

CAPTIONS = {
    "Fig1": ("ROH main effect on AD across model specifications.",
             "Odds ratios (Firth logistic regression; points) with Wald 95% CIs (lines) for carriers of the retained "
             "ROH versus non-carriers. Model 1 (bold, diamond) is the pre-specified primary model: AD ~ ROH + age + sex + "
             "PC1–5 + APOE ε4 dosage. Other rows show the original TRUROH estimates, alternative APOE codings, the same "
             "covariates on the APOE-typed subset (isolates the sample change), APOE strata, and adjustment for global "
             "ancestry and for local ancestry at the ROH locus. P values are penalized likelihood-ratio tests (Fisher "
             "exact for the 2×2 row), unadjusted for multiplicity; adjusted values are in the tables."),
    "Fig2": ("APOE ε4 and its interaction with ROH.",
             "a, APOE ε4 odds ratios from Model 1 (dosage) and alternative codings. b, ROH odds ratio at 0, 1 and 2 ε4 "
             "alleles from Model 2 (ROH × APOE ε4 dosage; delta-method 95% CIs), with ε4-stratified estimates for "
             "comparison; open markers/dashed CIs indicate APOE levels with no ROH carriers (model extrapolation); n = ROH "
             "carriers at that level. c, Covariate-standardized (g-computation) predicted probability of AD by ROH "
             "status and ε4 dosage from Model 2; intervals from 2,000 draws of the coefficient distribution; open markers "
             "mark cells with fewer than five observed individuals."),
    "Fig3": ("Ancestry and AD: global ancestry, local ancestry at the ROH locus and ancestry-specific effects.",
             "a, Odds ratio of AD per 10% global ancestry (squares) and per haplotype of local ancestry at the ROH locus "
             "conditional on global ancestry of the same component (filled circles) or without it (open circles); all "
             "models include age, sex, PC1–5 and APOE ε4. b, ROH odds ratio as a function of local ancestry dosage "
             "(ROH × local ancestry); solid line and band, observed carrier range; dashed, extrapolation. c, ROH odds ratio "
             "for carriers whose homozygous haplotype is of the indicated ancestral origin (versus non-carriers), from a "
             "model contrasting that origin with all other origins; P tests heterogeneity between origins. d, Per-allele "
             "APOE ε4 odds ratio across global ancestry (APOE × global ancestry). e, Per-allele APOE ε4 odds ratio by local "
             "ancestry at APOE (when supplied). f, ROH odds ratio across global ancestry (ROH × global ancestry)."),
    "FigS1": ("Data overview.",
              "a, Local ancestry dosage at the ROH locus among non-carriers (left bar) and ROH carriers (right bar); ROH "
              "carriers are almost always homozygous for a single ancestry. b, Local versus global ancestry for each "
              "component (r, Pearson correlation; carriers highlighted). c, Observed AD fraction (cases/N) by ε4 dosage "
              "and ROH status. d, ROH carriers by ancestral origin of the homozygous haplotype and AD status."),
    "FigS2": ("Every planned test.",
              "−log10 P for every converged test in each pre-defined family (jittered). Dashed lines: nominal P = 0.05 and "
              "Bonferroni threshold over all planned tests. Tests that could not be estimated are counted under each family "
              "and enter the multiplicity correction as P = 1."),
}


def _img(path, width=1900):
    try:
        from PIL import Image
        im = Image.open(path)
        if im.width > width:
            im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
    except Exception:  # noqa: BLE001
        data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _table(frame, cols, digits=3):
    cols = [c for c in cols if c in frame]
    if frame.empty or not cols:
        return "<p class='muted'>None.</p>"
    out = frame[cols].copy()
    for c in cols:
        if c in ("P_VALUE", "HOLM_P_FAMILY", "HOLM_P_ALL", "BH_Q_ALL", "WALD_P"):
            out[c] = out[c].map(lambda v: fmt_p_plain(v) if pd.notna(v) else "—")
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: f"{v:.{digits}g}" if pd.notna(v) else "—")
    return out.to_html(index=False, escape=True, na_rep="—", classes="t")


def _or_text(row):
    if row is None or not np.isfinite(row.get("OR", np.nan)):
        return "not estimable"
    return f"OR {row.OR:.2f} (95% CI {row.CI_LOW:.2f}–{row.CI_HIGH:.2f}; P = {fmt_p_plain(row.P_VALUE)})"


def _one(results, model, test, region):
    s = results[(results.MODEL == model) & (results.TEST == test) & (results.REGION_ID == region)]
    return s.iloc[0] if len(s) else None


def findings(results, coefs, decision, region, ancestries, audit):
    lines = []
    p1 = _one(results, "P1_ROH_cov_APOE", "ROH", region)
    a1 = _one(results, "P1_ROH_cov_APOE", "APOE4", region)
    p2 = _one(results, "P2_ROHxAPOE", "ROH:APOE4", region)
    a1o = _one(results, "A1_cov_original", "ROH", region)
    if p1 is not None:
        lines.append(f"<b>Model 1</b> (ROH + age + sex + PC1–5 + APOE ε4): ROH {_or_text(p1)}, N = {int(p1.N):,} "
                     f"({int(p1.CARRIER_CASES)} carrier cases / {int(p1.CARRIER_CONTROLS)} carrier controls). "
                     + (f"Without APOE (original M1 specification) ROH {_or_text(a1o)}." if a1o is not None else ""))
    if a1 is not None:
        lines.append(f"APOE ε4 (per allele) in Model 1: {_or_text(a1)}.")
    if p2 is not None:
        lines.append(f"<b>Model 2</b> ROH × APOE ε4 interaction: " + (
            f"ratio of ORs {p2.OR:.2f} (95% CI {p2.CI_LOW:.2f}–{p2.CI_HIGH:.2f}; P = {fmt_p_plain(p2.P_VALUE)}; "
            f"family-wise Holm P = {fmt_p_plain(p2.HOLM_P_FAMILY)})." if np.isfinite(p2.get("OR", np.nan))
            else str(p2.STATUS).replace("_", " ") + f" — {p2.DETAIL}"))
    k1, k5 = _one(results, "K1_P1_GLMM", "ROH", region), _one(results, "K5_P1_unrelated", "ROH", region)
    if k1 is not None and k1.CONVERGED == 1:
        lines.append(f"Accounting for kinship (GLMM, kinship variance τ = {k1.GLMM_TAU:.2f}): ROH {_or_text(k1)}"
                     + (f"; unrelated subset (N = {int(k5.N):,}, {int(k5.CARRIERS)} carriers): ROH {_or_text(k5)}." if k5 is not None and k5.CONVERGED == 1 else "."))
    anc = []
    for a in ancestries:
        for model, test, what in [(f"G1_{a}_adjust", f"Q10_{a}", f"global {a} (per 10%)"),
                                  (f"L1_{a}_local", f"LAD_{a}", f"local {a} at ROH locus (per haplotype, given global)"),
                                  (f"G2_{a}_APOExQ", f"APOE4:Q10_{a}", f"APOE ε4 × global {a}"),
                                  (f"L2_{a}_ROHxLA", f"ROH:LAD_{a}", f"ROH × local {a}"),
                                  (f"L5_{a}_origin", f"ROH_O_{a}", f"ROH haplotype of {a} origin vs other origins")]:
            row = _one(results, model, test, region)
            if row is not None and row.CONVERGED == 1 and np.isfinite(row.P_VALUE):
                anc.append((row.P_VALUE, what, row))
    if anc:
        anc.sort(key=lambda x: x[0])
        top = anc[:4]
        lines.append("Smallest ancestry-related P values (exploratory; see tables for all): " + "; ".join(
            f"{w}: OR {r.OR:.2f} (P = {fmt_p_plain(p)}, catalog-wide Holm P = {fmt_p_plain(r.HOLM_P_ALL)})" for p, w, r in top) + ".")
    return lines


def global_ancestry_section(results, coefs, decision, region, ancestries):
    rows = []
    for a in ancestries:
        w = coefs[(coefs.MODEL == f"L1_{a}_local") & (coefs.TERM == f"LAD_{a}") & (coefs.REGION_ID == region)] if not coefs.empty else coefs
        wo = coefs[(coefs.MODEL == f"L1n_{a}_local_noglobal") & (coefs.TERM == f"LAD_{a}") & (coefs.REGION_ID == region)] if not coefs.empty else coefs
        dd = decision[(decision.REGION_ID == region) & (decision.ANCESTRY == a)] if not decision.empty else decision
        rows.append(dict(ANCESTRY=a, CORR_LOCAL_GLOBAL=dd.CORR_LOCAL_GLOBAL.iloc[0] if len(dd) else np.nan,
                         R2_LOCAL_ON_GLOBAL_PCS=dd.R2_LOCAL_ON_GLOBAL_AND_PCS.iloc[0] if len(dd) else np.nan,
                         VIF_LOCAL=dd.VIF_LOCAL.iloc[0] if len(dd) else np.nan,
                         LOCAL_OR_WITH_GLOBAL=w.OR.iloc[0] if len(w) else np.nan,
                         LOCAL_OR_WITHOUT_GLOBAL=wo.OR.iloc[0] if len(wo) else np.nan,
                         PCT_CHANGE_LOG_OR=(100 * (np.log(wo.OR.iloc[0]) - np.log(w.OR.iloc[0])) / abs(np.log(w.OR.iloc[0]))
                                            if len(w) and len(wo) and abs(np.log(w.OR.iloc[0])) > 1e-6 else np.nan)))
    t = pd.DataFrame(rows)
    max_vif = float(t.VIF_LOCAL.max()) if len(t) and t.VIF_LOCAL.notna().any() else np.nan
    text = (
        "<p><b>Decision: yes — every primary local-ancestry model also contains global ancestry of the same component "
        "(plus PC1–5); the PC-only version is kept as a labelled sensitivity analysis.</b></p>"
        "<p>Local ancestry at a locus is a draw around an individual's genome-wide (global) ancestry. Without global "
        "ancestry, a local-ancestry coefficient mixes the locus-specific signal with every genome-wide correlate of "
        "ancestry (polygenic background, socio-environmental exposures, recruitment). Conditioning on global ancestry "
        "is the standard admixture-mapping contrast and isolates what is specific to this locus. The cost is "
        "collinearity, quantified below: the variance inflation of local ancestry"
        + (f" is at most {max_vif:.2f}" if np.isfinite(max_vif) else "") +
        ", which is modest when VIF &lt; 5. Global ancestry is entered one component at a time (or K−1 components "
        "with a reference) because the proportions sum to one. Because ROH carriers are homozygous for one ancestry, "
        "the ancestral origin of the ROH haplotype (Fig. 3c) is the most direct test of <i>which ancestry's</i> ROH "
        "drives an association.</p>")
    return text + _table(t, list(t.columns))


def relatedness_section(results, reldiag, audit, args):
    rel = audit.get("relatedness", {})
    kin_rows = results[results.FAMILY.eq("relatedness") | results.MODEL.isin(["P1_ROH_cov_APOE", "P2_ROHxAPOE"])]
    kin_rows = kin_rows[kin_rows.TEST.isin(["ROH", "ROH:APOE4", "FROH_PCT"])]
    has_matrix = "sha256" in rel
    head = (f"<p>Source: <code>{html.escape(str(rel.get('source')))}</code>. "
            + ("" if has_matrix else "<b>No kinship file was supplied</b>, so the GLMM models are reported as "
               "unavailable and the robust-SE / unrelated-subset analyses fall back to FID families. Supply "
               "<code>--kinship</code> (KING table or GCTA GRM) for the full relatedness analysis. ")
            + f"Primary family: <b>{'kinship GLMM' if args.primary == 'glmm' else 'Firth (no random effect)'}</b>.</p>"
            "<p>Kinship is a property of <i>pairs</i>, so it is not entered as an ordinary covariate. It is handled "
            "as (i) a GRM random effect (logistic mixed model), (ii) family-cluster-robust standard errors and (iii) a "
            "maximal unrelated subset. The individual-level relatedness quantity that <i>is</i> a covariate, the "
            "inbreeding coefficient (genome-wide F_ROH, focal region excluded), is added in K7/K8. Relatives who "
            "share the same IBD segment are not independent observations: if ROH carriers cluster in families, "
            "models that ignore kinship give P values that are too small. Compare the rows below; if they agree, "
            "relatedness does not drive the result.</p>")
    diag = _table(reldiag, [c for c in (reldiag.columns if reldiag is not None else []) if c != "source"]) if reldiag is not None else ""
    cols = ["REGION_ID", "MODEL", "LABEL", "TEST", "N", "CARRIERS", "OR", "CI_LOW", "CI_HIGH", "P_VALUE", "GLMM_TAU",
            "N_CLUSTERS", "CARRIER_CLUSTERS", "STATUS", "DETAIL"]
    return head + diag + _table(kin_rows, cols)


def make_report(output, results, coefs, contrasts, preds, diagnostics, decision, frames, audit, models, args, reldiag=None):
    output = Path(output)
    ancestries = audit["ancestries"]
    figures = []
    if not args.no_figures:
        figures = make_figures(output / "figures", results, coefs, contrasts, preds, frames, ancestries,
                               audit["apoe_locus_local_ancestry"]["available"], audit["cohort"])
    eligible = results.CONVERGED.eq(1)
    primary = results[results.FAMILY == "primary"]
    summary = dict(project=audit["project"], cohort=audit["cohort"], version=VERSION, status="completed",
                   regions=list(frames), planned_tests=len(results), converged_tests=int(eligible.sum()),
                   nominal_tests=int((eligible & results.P_VALUE.lt(.05)).sum()),
                   primary_tests=len(primary), primary_holm_tests=int((primary.CONVERGED.eq(1) & primary.HOLM_P_FAMILY.lt(.05)).sum()),
                   catalog_holm_tests=int((eligible & results.HOLM_P_ALL.lt(.05)).sum()),
                   apoe=dict((k, audit["apoe"].get(k)) for k in ("dosage_available_N", "carrier_available_N",
                                                                  "epsilon4_allele_frequency", "warnings", "mapping_file")),
                   figures={f"{tag}:{reg}": {k: str(v) for k, v in paths.items()} for tag, reg, paths in figures})
    write_json(output / "summary.json", summary)

    md = [f"# {audit['cohort']}: ROH × APOE × ancestry association (v{VERSION})", ""]
    blocks = []
    for region in frames:
        fl = findings(results, coefs, decision, region, ancestries, audit)
        md += [f"## {region}", ""] + ["- " + html.unescape(x.replace("<b>", "**").replace("</b>", "**")) for x in fl] + [""]
        blocks.append((region, fl))
    md += ["Primary family (Models 1 and 2) is Holm-corrected within family; every other test is exploratory and is also",
           "reported with a catalog-wide Holm correction. Cross-sectional case-control data describe association, not",
           "progression or causation."]
    (output / "interpretation.md").write_text("\n".join(md) + "\n")

    apoe = audit["apoe"]
    warn = "".join(f"<div class='warn'>⚠ {html.escape(w)}</div>" for w in apoe.get("warnings", []))
    if not apoe.get("dosage_available_N"):
        warn += ("<div class='warn'>⚠ APOE ε4 dosage is unavailable, so Models 1–2 and all APOE terms could not be "
                 "fitted. Supply the APOE map (profile <code>apoe_map</code> or <code>--apoe-map</code>) after checking "
                 "<code>apoe_source_value_counts.tsv</code>.</div>")
    fam_counts = results.groupby("FAMILY").agg(tests=("TEST", "size"), converged=("CONVERGED", lambda s: int((s == 1).sum())),
                                               nominal=("NOMINAL_P_LT_0_05", "sum")).reset_index()
    fam_counts["description"] = fam_counts.FAMILY.map(FAMILIES).fillna("Imported from the original TRUROH run")
    fig_html = []
    for i, (tag, region, paths) in enumerate(figures, 1):
        title, caption = CAPTIONS[tag]
        links = " · ".join(f"<a href='figures/{Path(p).name}'>{ext.upper()}</a>" for ext, p in paths.items())
        fig_html.append(f"<figure><img src='{_img(paths['png'])}' alt='{html.escape(title)}'><figcaption><b>{tag} — "
                        f"{html.escape(title)}</b>{' (' + html.escape(region) + ')' if region else ''} {html.escape(caption)}"
                        f"<div class='links'>{links}</div></figcaption></figure>")
    key_cols = ["REGION_ID", "MODEL", "LABEL", "TEST", "N", "CARRIER_CASES", "CARRIER_CONTROLS", "OR", "CI_LOW", "CI_HIGH",
                "P_VALUE", "HOLM_P_FAMILY", "HOLM_P_ALL", "STATUS", "DETAIL"]
    sections = []
    for region, fl in blocks:
        sections.append(f"<h3>{html.escape(region)}</h3><ul>" + "".join(f"<li>{x}</li>" for x in fl) + "</ul>")
    ga = "".join(global_ancestry_section(results, coefs, decision, r, ancestries) for r in frames)
    con = contrasts[contrasts.WITHIN_OBSERVED_RANGE.eq(1) & ~contrasts.GROUP.str.contains("_by_Q_|_by_LA_", regex=True)] if not contrasts.empty else contrasts
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ROH APOE Ancestry</title><style>
:root{{--ink:#1a1a1a;--muted:#5f6368;--line:#dde1e4;--accent:#0b5c86;--bg:#ffffff;--soft:#f4f6f7;--warn:#fff4e5}}
body{{font:15px/1.55 -apple-system,"Segoe UI",Arial,sans-serif;color:var(--ink);background:var(--bg);margin:0}}
main{{max-width:1080px;margin:0 auto;padding:28px 16px 64px}} h1{{font-size:26px;margin:0 0 4px;color:var(--accent)}}
h2{{font-size:19px;margin:34px 0 10px;border-bottom:1px solid var(--line);padding-bottom:4px}} h3{{font-size:16px;margin:18px 0 6px}}
.muted{{color:var(--muted)}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}}
.kpi{{background:var(--soft);border-radius:8px;padding:10px 12px}} .kpi b{{display:block;font-size:20px}} .kpi span{{font-size:12px;color:var(--muted)}}
.warn{{background:var(--warn);border-left:4px solid #d97706;padding:8px 12px;margin:8px 0;border-radius:4px}}
figure{{margin:22px 0;border:1px solid var(--line);border-radius:8px;padding:12px}} figure img{{width:100%;height:auto}}
figcaption{{font-size:13px;color:#333;margin-top:8px}} .links{{margin-top:4px;font-size:12px}}
table.t{{border-collapse:collapse;font-size:12px;margin:8px 0;display:block;overflow-x:auto;max-width:100%}}
table.t td,table.t th{{border:1px solid var(--line);padding:4px 6px;white-space:nowrap}} table.t th{{background:var(--soft);text-align:left}}
details{{margin:10px 0}} summary{{cursor:pointer;font-weight:600}} code{{background:var(--soft);padding:1px 4px;border-radius:3px}}
</style></head><body><main>
<h1>{html.escape(audit['cohort'])} — ROH × APOE × ancestry</h1>
<div class="muted">TRUROH association package v{VERSION} · source run signature {html.escape(str(audit['source_run_signature']))}</div>
<div class="grid">
<div class="kpi"><b>{audit['technical_N']:,}</b><span>genotyped</span></div>
<div class="kpi"><b>{audit['cases']:,} / {audit['controls']:,}</b><span>AD cases / controls</span></div>
<div class="kpi"><b>{audit['high_confidence_regions']}</b><span>retained ROH regions</span></div>
<div class="kpi"><b>{apoe.get('dosage_available_N', 0):,}</b><span>APOE ε4 dosage decoded</span></div>
<div class="kpi"><b>{(apoe.get('epsilon4_allele_frequency') or float('nan')):.3f}</b><span>ε4 allele frequency</span></div>
<div class="kpi"><b>{len(results)}</b><span>planned tests ({summary['converged_tests']} fitted)</span></div>
</div>{warn}
<h2>Key results</h2>{''.join(sections)}
<p class="muted">Primary hypotheses (Model 1 ROH effect, Model 2 ROH×APOE) are Holm-corrected within their family. Every
other test is exploratory: it is reported with its raw P, a within-family Holm P and a catalog-wide Holm P, and should be
described as such. Case-control data show association with AD status; they cannot establish effects on progression.</p>
<h2>Figures</h2>{''.join(fig_html) if fig_html else '<p class="muted">Figures disabled.</p>'}
<h2>Should global ancestry be included with local ancestry?</h2>{ga}
<h2>Relatedness (kinship)</h2>{relatedness_section(results, reldiag, audit, args)}
<h2>Primary models</h2>{_table(primary, key_cols)}
<h2>Effect contrasts (within observed support)</h2>
<p class="muted">Delta-method odds ratios for one variable at fixed levels of its modifier (e.g. ROH at 0/1/2 ε4 alleles).
Full curves, including extrapolated points, are in <code>tables/effect_contrasts.tsv</code>.</p>
{_table(con, ["REGION_ID", "MODEL", "GROUP", "MODIFIER", "VALUE", "N_AT_VALUE", "OR", "CI_LOW", "CI_HIGH", "WALD_P"])}
<h2>Test families</h2>{_table(fam_counts, ["FAMILY", "description", "tests", "converged", "nominal"])}
<details><summary>APOE, global and local ancestry tests</summary>{_table(results[results.FAMILY.isin(['apoe', 'global_ancestry', 'local_ancestry', 'apoe_locus_ancestry'])], key_cols)}</details>
<details><summary>Original TRUROH run (imported, unchanged)</summary>{_table(results[results.FAMILY == 'original_run'], key_cols)}</details>
<details><summary>APOE decoding audit</summary><p>Source <code>{html.escape(str(apoe.get('source_file')))}</code>, column
<code>{html.escape(str(apoe.get('source_column')))}</code>, map <code>{html.escape(str(apoe.get('mapping_file')))}</code>
({html.escape(str(apoe.get('mapping_encoding', 'none')))}). Matched {apoe.get('source_matched_N', 0)} of {audit['technical_N']} IDs.
Dosage counts {html.escape(str(apoe.get('dosage_counts', {})))}; mean ε4 dosage controls {apoe.get('mean_e4_dosage_controls', float('nan')):.3f}, cases {apoe.get('mean_e4_dosage_cases', float('nan')):.3f}.</p></details>
<details><summary>Carrier diagnostics</summary>{_table(diagnostics, list(diagnostics.columns))}</details>
<h2>Methods (draft text)</h2>
<p>Association between carrier status of each retained high-confidence ROH region and AD was estimated with Firth
bias-reduced logistic regression (penalized likelihood with Jeffreys prior) to accommodate the small number of carriers.
P values are penalized likelihood-ratio tests in which the tested coefficients are fixed at zero while the full-design
penalty is retained; confidence intervals are Wald intervals. The primary model (Model 1) adjusted for age (centred, per
decade), sex, the first five genetic principal components and APOE ε4 dosage (0/1/2); Model 2 added a ROH × ε4-dosage
product term. Age, sex and principal components were adjustment covariates only and were never modelled as effect
modifiers. Global ancestry proportions (per 10%) and local ancestry at the ROH locus (number of haplotypes, 0–2) were
analysed as covariates and as effect modifiers; local-ancestry models were adjusted for global ancestry of the same
component, with principal-component-only models as sensitivity analyses. Because ROH carriers are homozygous for a single
ancestry at the locus, we additionally contrasted carriers by the ancestral origin of the homozygous haplotype.
Ancestry-specific APOE effects were assessed with APOE × global ancestry terms{' and APOE × local ancestry at APOE' if audit['apoe_locus_local_ancestry']['available'] else ''}.
Relatedness was addressed with a logistic mixed model including a genetic relationship matrix random effect
(penalized quasi-likelihood with REML variance estimation and score tests), family-cluster-robust standard errors,
a maximal unrelated subset, and adjustment for genome-wide autozygosity (F<sub>ROH</sub>, excluding the tested region).
Covariate-standardized risks were obtained by g-computation. All tests were two-sided; the model catalog was fixed before
fitting; models lacking support (fewer than {args.min_carriers} carriers or non-carriers, or fewer than {args.min_interaction_cell}
carriers per modifier level for interactions) are reported as not estimable and enter multiplicity correction as P = 1.
The two primary tests were Holm-corrected within their family; all other tests are exploratory and are reported with
within-family and catalog-wide Holm corrections.</p>
<h2>Limitations</h2><ul>
<li>One retained region with few carriers: interaction and origin-specific estimates have wide intervals; non-estimable cells are reported, not dropped.</li>
<li>The GLMM uses PQL, which can underestimate the kinship variance for binary traits; its score tests are the GMMAT/SAIGE standard but are asymptotic.</li>
<li>Wald intervals and asymptotic LR P values are approximate with sparse cells.</li>
<li>Correction covers this catalog plus imported original tests, not earlier or later unrecorded analyses.</li>
<li>Associations from case-control status; no inference about disease progression or causality.</li></ul>
<h2>Files</h2><p><code>tables/all_association_tests.tsv</code> (every test), <code>tables/all_coefficients.tsv</code> (every term, natural units),
<code>tables/effect_contrasts.tsv</code>, <code>tables/predicted_risk.tsv</code>, <code>tables/ancestry_collinearity.tsv</code>,
<code>analysis_manifest.json</code> (catalog fixed before fitting), <code>apoe_audit.json</code>, <code>figures/</code> (PDF, SVG, 600-dpi PNG/TIFF).</p>
</main></body></html>"""
    (output / "report.html").write_text(page)
    return summary
