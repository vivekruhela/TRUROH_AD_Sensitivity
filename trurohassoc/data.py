"""Load a completed TRUROH run (read-only) and build per-region analysis frames."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .apoe import prepare_apoe
from .relatedness import froh_excluding, load_relatedness, read_all_segments, read_inbreeding
from .common import (ALIASES, check_proportions, collapse_ancestry, numeric, read_json, read_tsv, sha256,
                     unique, validate_ids, write_json, write_tsv)

NUMERIC_PHENO = ["AD", "AGE", "SEX", "APOE4", "PC1", "PC2", "PC3", "PC4", "PC5"]
REQUIRED = ["run_manifest.json", "final_status.json", "config.resolved.json",
            "inputs/phenotypes_prepared.tsv", "results/regions_scored.tsv", "results/members.tsv.gz"]
OPTIONAL = ["results/global_ancestry.tsv", "results/local_ancestry_for_association.tsv.gz", "results/associations.tsv"]


PRIMARY_CLOSE_MARGIN = 0.05   # top-two mean global ancestry closer than this -> "auto" choice is flagged as a close call


def resolve_primary_ancestry(requested, ancestries, pheno, root):
    """Primary ancestry of the analysis: listed first in the catalog, tables and report.

    Explicit value: used as given (a pre-specified hypothesis); a note is recorded when it is not the cohort-majority
    ancestry. "auto": the ancestry with the highest mean global ancestry (Q_K) over the prepared cohort; never uses
    the outcome. Without global ancestry, "auto" uses the source run's results/primary_ancestry_selection.json
    (TRUROH >= 2.2.0), else the first configured ancestry. Returns (ancestry, info)."""
    auto = str(requested).strip().casefold() == "auto"
    configured = "auto" if auto else ALIASES.get(str(requested).strip(), str(requested).strip())
    if not auto and configured not in ancestries:
        raise ValueError(f'primary ancestry must be "auto" or one of {ancestries}, not {requested!r}')
    info = {"configured": configured, "method": "auto" if auto else "explicit", "candidates": list(ancestries),
            "statistic": "mean global ancestry over the prepared cohort", "close_margin": PRIMARY_CLOSE_MARGIN,
            "notes": []}
    means = {a: float(pheno["Q_" + a].mean()) for a in ancestries if "Q_" + a in pheno}
    if not means:
        selected, source = configured, None
        if auto:
            run_choice = root / "results/primary_ancestry_selection.json"
            selected = read_json(run_choice).get("selected") if run_choice.is_file() else None
            if selected in ancestries:
                source = str(run_choice)
            else:
                selected = ancestries[0]
                info["notes"].append(f"global ancestry unavailable; using the first configured ancestry {selected}")
        info.update(selected=selected, source=source, majority=None, mean_global_ancestry=None)
        return selected, info
    ranked = sorted(((v, a) for a, v in means.items()), reverse=True)
    majority = ranked[0][1]
    margin = ranked[0][0] - ranked[1][0] if len(ranked) > 1 else None
    selected = majority if auto else configured
    if auto and margin is not None and margin < PRIMARY_CLOSE_MARGIN:
        info["notes"].append(f"close call: {ranked[0][1]} {ranked[0][0]:.3f} vs {ranked[1][1]} {ranked[1][0]:.3f}; "
                             "consider setting the primary ancestry explicitly")
    if not auto and configured != majority:
        info["notes"].append(f"explicit primary ancestry {configured} ({means.get(configured, np.nan):.3f}) is not the "
                             f"cohort-majority ancestry {majority} ({ranked[0][0]:.3f}); kept as the pre-specified hypothesis")
    info.update(selected=selected, source="results/global_ancestry.tsv", majority=majority,
                top_two_margin=None if margin is None else round(margin, 4),
                mean_global_ancestry={a: round(v, 4) for v, a in ranked})
    for note in info["notes"]:
        print("NOTE primary ancestry: " + note, flush=True)
    return selected, info


def load_run(root, output, apoe_map=None, apoe_overrides=None, apoe_locus_la=None, kinship=None, inbreeding=None,
             primary_ancestry=None):
    root = Path(root)
    for path in REQUIRED:
        if not (root / path).is_file():
            raise FileNotFoundError(root / path)
    manifest = read_json(root / "run_manifest.json")
    final = read_json(root / "final_status.json")
    if final.get("state") != "completed" or final.get("run_signature") != manifest.get("run_signature"):
        raise ValueError("A completed, internally consistent TRUROH run is required")
    cfg = read_json(root / "config.resolved.json")
    if cfg.get("project") != manifest.get("project"):
        raise ValueError("Saved configuration and manifest have different projects")
    pheno = read_tsv(root / "inputs/phenotypes_prepared.tsv")
    validate_ids(pheno, "Prepared phenotype")
    if len(pheno) != int(manifest["technical_N"]):
        raise ValueError("Prepared phenotype sample count disagrees with run manifest")
    columns = cfg.get("columns", {}) or {}
    core = cfg.get("association", {}).get("covariates", ["AGE", "SEX", "PC1", "PC2", "PC3", "PC4", "PC5"])
    aliases = {c.casefold(): c for c in NUMERIC_PHENO}
    aliases.update({columns.get("age", "AGE").casefold(): "AGE", columns.get("sex", "SEX").casefold(): "SEX"})
    core = unique([aliases.get(c.casefold(), c) for c in core])
    if not set(["AD", "AGE", "SEX", "FID", "IID"] + core).issubset(pheno):
        raise ValueError("Prepared phenotype lacks required standardized variables")
    numeric(pheno, unique(NUMERIC_PHENO + core))
    if not pheno.AD.dropna().isin([0, 1]).all() or not pheno.SEX.dropna().isin([0, 1]).all():
        raise ValueError("Prepared AD and SEX must use the saved 0/1 coding")
    pheno, apoe_audit = prepare_apoe(pheno, cfg, output, apoe_map, apoe_overrides)
    age_center = float(pheno.AGE.mean()) if pheno.AGE.notna().any() else np.nan
    pheno["AGE_C"] = (pheno.AGE - age_center) / 10.0          # per decade
    # Covariate names used in models: AGE -> AGE_C (centred, per 10 years)
    core = unique(["AGE_C" if x == "AGE" else x for x in core])

    scored = read_tsv(root / "results/regions_scored.tsv")
    if scored.REGION_ID.duplicated().any():
        raise ValueError("Duplicate region IDs in scored results")
    numeric(scored, ["HIGH_CONFIDENCE", "CHR", "CONS_START", "CONS_END", "N_CARRIERS"])
    if not scored.HIGH_CONFIDENCE.isin([0, 1]).all():
        raise ValueError("Invalid HIGH_CONFIDENCE flags")
    hc = scored.loc[scored.HIGH_CONFIDENCE.eq(1)].copy().sort_values(["CHR", "CONS_START"])
    hc_ids = set(hc.REGION_ID)
    pieces = []
    for chunk in pd.read_csv(root / "results/members.tsv.gz", sep="\t", dtype=str, keep_default_na=False,
                             usecols=["REGION_ID", "IID"], chunksize=250000):
        pieces.append(chunk[chunk.REGION_ID.isin(hc_ids)])
    members = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["REGION_ID", "IID"])
    if members.duplicated(["REGION_ID", "IID"]).any() or not set(members.IID).issubset(set(pheno.IID)):
        raise ValueError("Invalid or duplicated high-confidence region membership")
    for region in hc.itertuples(index=False):
        if int(members.REGION_ID.eq(region.REGION_ID).sum()) != int(region.N_CARRIERS):
            raise ValueError(f"Carrier count disagreement for {region.REGION_ID}")

    # ---- global ancestry (Q): kept on the 0-1 scale and also as Q10_K = per 10 percentage points
    ancestries = [ALIASES.get(x, x) for x in cfg.get("association", {}).get("ancestries", ["NAM", "EUR", "AFR"])]
    reference = ALIASES.get(cfg.get("association", {}).get("global_reference", "EUR"), "EUR")
    qpath = root / OPTIONAL[0]
    q_available = []
    if qpath.is_file():
        q = collapse_ancestry(read_tsv(qpath), "Q_")
        validate_ids(q, "Global ancestry")
        if set(q.IID) != set(pheno.IID):
            raise ValueError("Global ancestry does not cover exactly the prepared cohort")
        all_q = [c for c in q if c.startswith("Q_")]
        check_proportions(q, all_q, "global")
        for c in all_q:
            q["Q10_" + c[2:]] = 10.0 * q[c]
        q_available = [c[2:] for c in all_q]
        pheno = pheno.merge(q, on="IID", validate="one_to_one", how="left")
    if q_available and reference not in q_available:
        reference = q_available[-1]
    ancestries = [a for a in ancestries if (not q_available) or a in q_available] or q_available
    primary_ancestry, primary_info = resolve_primary_ancestry(
        primary_ancestry or cfg.get("association", {}).get("primary_ancestry") or "auto", ancestries, pheno, root)
    ancestries = [primary_ancestry] + [a for a in ancestries if a != primary_ancestry]

    # ---- optional local ancestry at the APOE locus (chr19:44.9 Mb), e.g. from RFMix .msp
    apoe_la_info = {"file": None, "available": False}
    if apoe_locus_la:
        la = read_tsv(apoe_locus_la)
        if "IID" not in la or not any(c.startswith("LA_") for c in la):
            raise ValueError(f"--apoe-locus-la {apoe_locus_la} is not a local-ancestry-at-APOE table: it needs an IID "
                             f"column and LA_<ancestry> columns, but has {la.columns.tolist()[:8]}. Build it from the "
                             "chr19 RFMix .msp.tsv with tools/make_apoe_locus_la.py (an RFMix sample map is the wrong file).")
        la = collapse_ancestry(la, "LA_")
        validate_ids(la, "APOE-locus local ancestry")
        cols = [c for c in la if c.startswith("LA_")]
        la = la[la.IID.isin(set(pheno.IID))].copy()
        complete = la[np.isfinite(la[cols].to_numpy(float)).all(axis=1)]
        check_proportions(complete, cols, "APOE-locus local")
        for c in cols:
            la["LAA_" + c[3:]] = 2.0 * la[c]       # haplotype dosage 0..2 at APOE
        pheno = pheno.merge(la[["IID"] + ["LAA_" + c[3:] for c in cols]], on="IID", how="left", validate="one_to_one")
        apoe_la_info = {"file": str(Path(apoe_locus_la).resolve()), "sha256": sha256(apoe_locus_la), "available": True,
                        "matched_N": int(len(complete))}

    local = read_tsv(root / OPTIONAL[1]) if (root / OPTIONAL[1]).is_file() else pd.DataFrame(columns=["REGION_ID", "IID"])
    if local.duplicated(["REGION_ID", "IID"]).any():
        raise ValueError("Duplicate per-region local ancestry IDs")
    original = read_tsv(root / OPTIONAL[2]) if (root / OPTIONAL[2]).is_file() else pd.DataFrame()

    # ---- relatedness and genome-wide autozygosity
    rel, rel_summary = load_relatedness(kinship, pheno)
    extras = {"relatedness": rel, "segments": None, "inbreeding": None}
    if inbreeding:
        extras["inbreeding"] = read_inbreeding(inbreeding, pheno.IID)
        froh_info = {"source": str(Path(inbreeding).resolve()), "sha256": sha256(inbreeding),
                     "matched_N": int(extras["inbreeding"].notna().sum())}
    else:
        extras["segments"] = read_all_segments(root, scored)
        froh_info = {"source": "union of all TRUROH candidate ROH regions carried, focal region excluded",
                     "segments": len(extras["segments"])}
    inputs = {p: {"bytes": (root / p).stat().st_size, "sha256": sha256(root / p)} for p in REQUIRED + OPTIONAL if (root / p).is_file()}
    audit = dict(project=manifest["project"], cohort=manifest.get("cohort", cfg.get("cohort_label", cfg["project"])),
                 source_run=root, source_run_signature=manifest["run_signature"], technical_N=len(pheno),
                 outcome_known_N=int(pheno.AD.notna().sum()), cases=int(pheno.AD.eq(1).sum()),
                 controls=int(pheno.AD.eq(0).sum()), high_confidence_regions=len(hc), candidate_regions=len(scored),
                 configured_covariates=core, age_center_years=age_center, age_unit_years=10,
                 global_reference=reference, ancestries=ancestries, global_ancestry_available=bool(q_available),
                 primary_ancestry=primary_ancestry, primary_ancestry_selection=primary_info,
                 local_ancestry_rows=len(local), apoe=apoe_audit, apoe_locus_local_ancestry=apoe_la_info,
                 relatedness=rel_summary, froh=froh_info, locked=manifest.get("locked"), input_files=inputs)
    write_json(output / "input_audit.json", audit)
    write_tsv(output / "retained_regions.tsv", hc)
    return cfg, pheno, core, hc, members, local, original, audit, extras


def region_frame(pheno, region, members, local, ancestries, extras=None):
    """Adds ROH carrier status, local-ancestry dosage at the region (LAD_K, 0-2 haplotypes) and the
    ancestral origin of each carrier's homozygous haplotype (ROH_O_K)."""
    data = pheno.copy()
    carriers = set(members.loc[members.REGION_ID.eq(region.REGION_ID), "IID"])
    data["ROH"] = data.IID.isin(carriers).astype(float)
    current = local.loc[local.REGION_ID.eq(region.REGION_ID)].copy()
    if not current.empty:
        if set(current.IID) != set(data.IID):
            raise ValueError(f"Local ancestry sample mismatch for {region.REGION_ID}")
        if "ROH" in current:
            expected = current.IID.isin(carriers).astype(int)
            if not pd.to_numeric(current.ROH, errors="coerce").eq(expected).all():
                raise ValueError("Local ancestry carrier labels disagree with membership")
        collapsed = collapse_ancestry(current, "LA_")
        columns = [c for c in collapsed if c.startswith("LA_")]
        check_proportions(collapsed, columns, "local")
        for c in columns:
            collapsed["LAD_" + c[3:]] = 2.0 * collapsed[c]
        data = data.merge(collapsed, on="IID", how="left", validate="one_to_one")
        present = [a for a in ancestries if "LA_" + a in data]
        if present:
            props = data[["LA_" + a for a in present]].to_numpy(float)
            origin = np.array(present)[np.argmax(props, axis=1)]
            data["ROH_ORIGIN"] = np.where(data.ROH.eq(1), origin, "none")
            data["ROH_ORIGIN_PURITY"] = np.where(data.ROH.eq(1), props.max(axis=1), np.nan)
            for a in present:
                data["ROH_O_" + a] = (data.ROH.eq(1) & (data.ROH_ORIGIN == a)).astype(float)
    if extras:
        if extras.get("inbreeding") is not None:
            data["FROH_PCT"] = data.IID.map(extras["inbreeding"]).astype(float)
        elif extras.get("segments") is not None:
            data["FROH_PCT"] = froh_excluding(extras["segments"], data.IID, int(region.CHR), int(region.CONS_START),
                                              int(region.CONS_END)).to_numpy()
        rel = extras.get("relatedness")
        if rel is not None:
            data["CLUSTER"] = data.IID.map(rel.cluster)
            core_cols = [c for c in ["AD", "AGE", "SEX", "APOE4"] if c in data]
            complete = data[core_cols].notna().all(axis=1) & data.AD.notna()
            prefer = dict(zip(data.IID, (data.ROH.eq(1) * 2 + complete.astype(int)).astype(int)))
            usable = data.loc[data.AD.notna(), "IID"]
            keep = rel.unrelated(list(usable), {k: v for k, v in prefer.items()})
            data["UNRELATED"] = data.IID.isin(keep).astype(float)
            data.loc[data.AD.isna(), "UNRELATED"] = np.nan
    return data


def relatedness_diagnostics(data, region_id, rel):
    """How ROH carriers are related to each other - the reason kinship matters for a shared IBD segment."""
    carriers = set(data.loc[data.ROH.eq(1) & data.AD.notna(), "IID"])
    with_carrier_relative = sum(1 for c in carriers if rel.related_edges.get(c, set()) & carriers)
    clusters = pd.Series([rel.cluster[c] for c in carriers])
    summary = rel.summary(list(data.loc[data.AD.notna(), "IID"]))
    summary.update(REGION_ID=region_id, carriers=len(carriers),
                   carriers_with_carrier_relative=int(with_carrier_relative),
                   independent_carrier_clusters=int(clusters.nunique()),
                   largest_carrier_cluster=int(clusters.value_counts().max()) if len(clusters) else 0,
                   unrelated_subset_N=int(data.UNRELATED.eq(1).sum()) if "UNRELATED" in data else None,
                   unrelated_subset_carriers=int((data.UNRELATED.eq(1) & data.ROH.eq(1)).sum()) if "UNRELATED" in data else None)
    return summary


def carrier_diagnostics(data, region_id, ancestries):
    rows = []
    groups = [("all", data)]
    for col in ["SEX", "APOE4", "APOE_CARRIER"]:
        if col in data:
            for value in sorted(data[col].dropna().unique()):
                groups.append((f"{col}={value:g}", data[data[col].eq(value)]))
    if "ROH_ORIGIN" in data:
        for a in ancestries:
            groups.append((f"ROH_origin={a}", data[(data.ROH_ORIGIN == a) | data.ROH.eq(0)]))
    for name, subset in groups:
        for outcome in (0, 1):
            part = subset[subset.AD.eq(outcome)]
            carriers = part[part.ROH.eq(1)]
            row = dict(REGION_ID=region_id, STRATUM=name, AD=outcome, N=len(part), CARRIERS=len(carriers),
                       NONCARRIERS=int(part.ROH.eq(0).sum()))
            for a in ancestries:
                if "LAD_" + a in data:
                    row[f"LAD_{a}_MEAN_CARRIERS"] = carriers["LAD_" + a].mean()
                    row[f"LAD_{a}_MEAN_NONCARRIERS"] = part.loc[part.ROH.eq(0), "LAD_" + a].mean()
            rows.append(row)
    return rows


def ancestry_decision(data, region_id, ancestries, core):
    """Quantifies why/whether global ancestry must accompany local ancestry."""
    rows = []
    pcs = [c for c in core if c.startswith("PC")]
    for a in ancestries:
        la, q = "LAD_" + a, "Q10_" + a
        if la not in data or q not in data:
            continue
        d = data[[la, q] + pcs].dropna()
        r_lq = float(np.corrcoef(d[la], d[q])[0, 1]) if len(d) > 2 else np.nan
        r_pc = max((abs(float(np.corrcoef(d[q], d[p])[0, 1])) for p in pcs), default=np.nan)
        # variance of local ancestry explained by global ancestry + PCs
        X = np.column_stack([np.ones(len(d)), d[[q] + pcs].to_numpy(float)])
        fitted = X @ np.linalg.lstsq(X, d[la].to_numpy(float), rcond=None)[0]
        r2 = 1 - np.var(d[la] - fitted) / np.var(d[la]) if np.var(d[la]) > 0 else np.nan
        rows.append(dict(REGION_ID=region_id, ANCESTRY=a, CORR_LOCAL_GLOBAL=r_lq, MAX_ABS_CORR_GLOBAL_PC=r_pc,
                         R2_LOCAL_ON_GLOBAL_AND_PCS=r2, VIF_LOCAL=1 / (1 - r2) if r2 < 1 else np.inf))
    return rows
