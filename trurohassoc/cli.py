"""End-to-end driver: profile -> audit -> fixed catalog -> fits -> multiplicity -> figures -> report."""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import VERSION
from .catalog import FAMILIES, model_catalog
from .common import read_json, sha256, write_json, write_tsv
from .data import ancestry_decision, carrier_diagnostics, load_run, region_frame, relatedness_diagnostics
from .firth import adjust_all_tests
from .fit import fit_model, formula, original_test_rows

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# Only these profile values are file paths; column names (apoe_column, ...) stay plain strings.
PATH_KEYS = {"config", "run_dir", "out_dir", "out_parent", "apoe_map", "apoe_source", "apoe_locus_la", "kinship", "inbreeding"}
PROFILE_KEYS = {"config", "run_dir", "out_dir", "out_parent", "out_name", "apoe_map", "apoe_column", "apoe_source",
                "apoe_fid_column", "apoe_iid_column", "apoe_locus_la", "min_carriers", "min_interaction_cell",
                "kinship", "inbreeding", "primary",
                "description"}


def resolve(value):
    if value is None:
        return None
    text = str(value)
    if text.startswith("@pkg/"):
        return (PACKAGE_ROOT / text[5:]).resolve()
    return Path(os.path.expandvars(text)).expanduser()


def parse(argv=None):
    p = argparse.ArgumentParser(description="ROH x APOE x ancestry association analysis of a completed TRUROH run")
    p.add_argument("--profile", help="Cohort profile name (profiles/<name>.json) or path; CLI flags override it")
    p.add_argument("--config", type=Path, help="Cohort JSON used to locate output_root of the completed run")
    p.add_argument("--run-dir", type=Path, help="Completed TRUROH output directory (alternative to --config)")
    p.add_argument("--out-dir", type=Path, help="Exact new output directory (must not exist)")
    p.add_argument("--out-parent", type=Path, help="Parent directory; a timestamped subdirectory is created")
    p.add_argument("--audit-only", action="store_true", help="Write audits and model plan; no fits")
    p.add_argument("--apoe-map", type=Path, help="Explicit APOE code map (epsilon4_dosage or epsilon4_carrier)")
    p.add_argument("--apoe-column", help="APOE source header (e.g. APOE_reported)")
    p.add_argument("--apoe-source", type=Path, help="APOE source TSV (default: saved phenotype_file)")
    p.add_argument("--apoe-fid-column")
    p.add_argument("--apoe-iid-column")
    p.add_argument("--apoe-locus-la", type=Path, help="Optional TSV: IID + LA_<ancestry> proportions at APOE (chr19)")
    p.add_argument("--kinship", type=Path, help="KING kinship table (.kin0 / plink2 --make-king-table) or GCTA GRM prefix")
    p.add_argument("--inbreeding", type=Path, help="Optional TSV IID + F (0-1) to replace the F_ROH computed from TRUROH regions")
    p.add_argument("--primary", choices=["firth", "glmm"], help="Primary family: Firth Models 1-2 (default) or their kinship GLMM versions")
    p.add_argument("--min-carriers", type=int)
    p.add_argument("--min-interaction-cell", type=int)
    p.add_argument("--max-fit-seconds", type=float, default=120)
    p.add_argument("--max-iterations", type=int, default=1000)
    p.add_argument("--tolerance", type=float, default=1e-5)
    p.add_argument("--prediction-draws", type=int, default=2000)
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)
    args.profile_data = {}
    if args.profile:
        path = Path(args.profile)
        if not path.suffix:
            path = PACKAGE_ROOT / "profiles" / f"{args.profile}.json"
        prof = read_json(path)
        unknown = set(prof) - PROFILE_KEYS
        if unknown:
            p.error(f"Unknown profile keys {sorted(unknown)} in {path}")
        args.profile_data = dict(prof, _path=str(path))
        for key, value in prof.items():
            if key in ("description", "out_name"):
                continue
            if getattr(args, key, None) is None:
                setattr(args, key, resolve(value) if key in PATH_KEYS and isinstance(value, str) else value)
    args.min_carriers = args.min_carriers or 3
    args.primary = args.primary or "firth"
    if args.primary == "glmm" and not args.kinship:
        p.error("--primary glmm requires --kinship")
    args.min_interaction_cell = args.min_interaction_cell or 5
    if not args.config and not args.run_dir:
        p.error("Give --profile, --config or --run-dir")
    if args.config and args.run_dir:
        p.error("Use only one of --config / --run-dir")
    overrides = {k: v for k, v in {"source": args.apoe_source, "column": args.apoe_column,
                                   "fid_column": args.apoe_fid_column, "iid_column": args.apoe_iid_column}.items() if v}
    if overrides and not args.apoe_map:
        p.error("APOE source overrides require --apoe-map with documented value encoding")
    args.apoe_overrides = overrides
    if min(args.min_carriers, args.min_interaction_cell, args.max_iterations, args.prediction_draws) < 1 or \
            args.max_fit_seconds <= 0 or args.tolerance <= 0:
        p.error("Fit limits and support counts must be positive")
    return args


def locate(args):
    if args.config:
        supplied = read_json(args.config)
        root = Path(supplied["output_root"]).expanduser()
        if not root.is_absolute():
            root = args.config.resolve().parent / root
    else:
        root = args.run_dir.expanduser()
    root = root.resolve()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + str(os.getpid())
    name = (args.profile_data.get("out_name") or root.name + "_ROH_APOE_ancestry") + "_" + stamp
    if args.out_dir:
        output = args.out_dir
    elif args.out_parent:
        output = args.out_parent / name
    else:
        output = root.with_name(name)
    output = output.expanduser().resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Use a separate output directory outside the completed TRUROH run")
    return root, output


def main(argv=None):
    args = parse(argv)
    root, output = locate(args)
    output.mkdir(parents=True, exist_ok=False)
    (output / "tables").mkdir()
    print(f"TRUROH ROH-APOE-ancestry v{VERSION}\nInput completed run: {root}\nOutput: {output}", flush=True)
    write_json(output / "analysis_status.json", {"state": "running", "version": VERSION, "source_run": root})
    try:
        return run(args, root, output)
    except Exception as exc:
        write_json(output / "analysis_status.json", {"state": "failed", "version": VERSION, "error": str(exc)})
        raise


def run(args, root, output):
    cfg, pheno, core, hc, members, local, original, audit, extras = load_run(
        root, output, args.apoe_map, args.apoe_overrides, args.apoe_locus_la, args.kinship, args.inbreeding)
    ancestries, reference = audit["ancestries"], audit["global_reference"]
    rel = extras["relatedness"]
    models = model_catalog(core, ancestries, reference, apoe_locus=audit["apoe_locus_local_ancestry"]["available"],
                           primary=args.primary)
    plan, frames, diagnostics, decision, reldiag = [], {}, [], [], []
    for region in hc.itertuples(index=False):
        data = region_frame(pheno, region, members, local, ancestries, extras)
        reldiag.append(relatedness_diagnostics(data, region.REGION_ID, rel))
        frames[region.REGION_ID] = data
        diagnostics.extend(carrier_diagnostics(data, region.REGION_ID, ancestries))
        decision.extend(ancestry_decision(data, region.REGION_ID, ancestries, core))
        for model in models:
            for name, terms in model.tests.items():
                plan.append(dict(REGION_ID=region.REGION_ID, FAMILY=model.test_family.get(name, model.family), MODEL=model.name, LABEL=model.label,
                                 TEST=name, TESTED_TERMS=";".join(terms), DF=len(terms), FORMULA=formula(model),
                                 SAMPLE_RULE=model.sample, METHOD=model.method))
    plan = pd.DataFrame(plan)
    diagnostics, decision = pd.DataFrame(diagnostics), pd.DataFrame(decision)
    write_tsv(output / "tables/model_plan.tsv", plan)
    write_tsv(output / "tables/carrier_diagnostics.tsv", diagnostics)
    write_tsv(output / "tables/ancestry_collinearity.tsv", decision)
    reldiag = pd.DataFrame(reldiag)
    write_tsv(output / "tables/relatedness_diagnostics.tsv", reldiag)
    manifest = dict(version=VERSION, script_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                    created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    profile=args.profile_data, arguments={k: v for k, v in vars(args).items() if k != "profile_data"},
                    families=FAMILIES, model_catalog=[asdict(m) for m in models], planned_test_count=len(plan),
                    timing="Catalog fixed before fitting; written before any model is fitted",
                    exclusions="Complete cases and fixed support rules only; nothing selected by p-value")
    write_json(output / "analysis_manifest.json", manifest)
    print(f"Regions: {len(hc)}; catalog: {len(models)} models / {sum(len(m.tests) for m in models)} tests per region", flush=True)
    print(f"APOE dosage N={audit['apoe']['dosage_available_N']}; carrier N={audit['apoe']['carrier_available_N']}; "
          f"ancestries={ancestries} (reference {reference})", flush=True)
    print("Relatedness: " + "; ".join(f"{k}={v}" for k, v in audit["relatedness"].items() if k not in ("sha256",)), flush=True)
    if args.audit_only:
        write_json(output / "analysis_status.json", {"state": "audit_completed", "version": VERSION, "fits_performed": 0})
        print("Audit only: check apoe_source_value_counts.tsv / apoe_audit.json before fitting.", flush=True)
        return 0

    rows = original_test_rows(original, set(hc.REGION_ID))
    coefs, contrasts, preds = [], [], []
    rng = np.random.default_rng(20261008)
    for region in hc.itertuples(index=False):
        for index, model in enumerate(models, 1):
            print(f"{region.REGION_ID} [{index}/{len(models)}] {model.name}", flush=True)
            r, c, k, p = fit_model(frames[region.REGION_ID], model, core, region, args, rng, rel)
            rows.extend(r); coefs.extend(c); contrasts.extend(k); preds.extend(p)
        write_tsv(output / "tables/association_tests.partial.tsv", pd.DataFrame(rows))
    results = pd.DataFrame(rows)
    for key in ["P_VALUE", "CONVERGED", "OR", "CI_LOW", "CI_HIGH"]:
        if key not in results:
            results[key] = np.nan
        results[key] = pd.to_numeric(results[key], errors="coerce")
    results.loc[results.CONVERGED.ne(1), "P_VALUE"] = np.nan
    adj = adjust_all_tests(results.P_VALUE)
    results["HOLM_P_ALL"], results["BONFERRONI_P_ALL"], results["BH_Q_ALL"] = adj["HOLM"], adj["BONFERRONI"], adj["BH"]
    results["HOLM_P_FAMILY"] = np.nan
    for family, idx in results.groupby("FAMILY").groups.items():
        results.loc[idx, "HOLM_P_FAMILY"] = adjust_all_tests(results.loc[idx, "P_VALUE"])["HOLM"]
    results["N_TESTS_ALL"] = len(results)
    results["NOMINAL_P_LT_0_05"] = (results.CONVERGED.eq(1) & results.P_VALUE.lt(.05)).astype(int)
    tables = output / "tables"
    write_tsv(tables / "all_association_tests.tsv", results)
    write_tsv(tables / "all_coefficients.tsv", pd.DataFrame(coefs))
    write_tsv(tables / "effect_contrasts.tsv", pd.DataFrame(contrasts))
    write_tsv(tables / "predicted_risk.tsv", pd.DataFrame(preds))
    (tables / "association_tests.partial.tsv").unlink(missing_ok=True)

    from .report import make_report
    summary = make_report(output, results, pd.DataFrame(coefs), pd.DataFrame(contrasts), pd.DataFrame(preds),
                          diagnostics, decision, frames, audit, models, args, reldiag)
    write_json(output / "analysis_status.json", {"state": "completed", "version": VERSION,
                                                 "planned_test_count": len(results), "source_run": root})
    print(f"Completed: {len(results)} tests; {summary['converged_tests']} converged; {summary['nominal_tests']} nominal p<0.05; "
          f"{summary['primary_holm_tests']} primary Holm<0.05.\nReport: {output / 'report.html'}", flush=True)
    return 0


def entry():
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
