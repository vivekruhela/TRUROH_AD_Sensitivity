#!/usr/bin/env python3
"""Assertions for tests/e2e/run_e2e.sh."""
import argparse
import json
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", type=Path, required=True)
ap.add_argument("--audit", type=Path, required=True)
ap.add_argument("--sbatch-log", type=Path, required=True)
ap.add_argument("--kinship", default="")
ap.add_argument("--apoe-la", default="")
a = ap.parse_args()
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


audit = json.loads((a.audit / "analysis_status.json").read_text())
check(audit.get("state") == "audit_completed" and audit.get("fits_performed") == 0, "audit-only run fitted nothing")
check((a.audit / "apoe_source_value_counts.tsv").is_file(), "APOE source value audit written")

calls = [json.loads(line) for line in a.sbatch_log.read_text().splitlines()]
check(len(calls) == 1, "exactly one sbatch submission")
opts = calls[0]["options"] if calls else {}
check(not any(k in opts for k in ("constraint", "partition", "account")), "empty partition/account/constraint omitted")
check(opts.get("cpus-per-task") == "2" and opts.get("mem") == "4G", "site.env resources applied (2 CPUs, 4G)")
check(bool(calls) and all(t["rc"] == 0 for t in calls[0]["tasks"]), "emulated SLURM job exited 0")

outs = sorted(p for p in a.results.glob("E2E_AD_*") if p.is_dir())
check(len(outs) == 1, "one time-stamped result folder under out_parent")
if outs:
    out = outs[0]
    status = json.loads((out / "analysis_status.json").read_text())
    check(status.get("state") == "completed", "analysis_status.state == completed")
    check((out / "report.html").stat().st_size > 50_000, "self-contained HTML report written")
    for f in ["interpretation.md", "summary.json", "apoe_audit.json", "analysis_manifest.json", "input_audit.json"]:
        check((out / f).is_file(), f"{f} written")
    tests = pd.read_csv(out / "tables/all_association_tests.tsv", sep="\t")
    plan = pd.read_csv(out / "tables/model_plan.tsv", sep="\t")
    check(len(tests) >= len(plan), "every planned test reported (including not estimable)")
    for t in ["all_coefficients.tsv", "effect_contrasts.tsv", "predicted_risk.tsv", "ancestry_collinearity.tsv",
              "carrier_diagnostics.tsv", "relatedness_diagnostics.tsv"]:
        check((out / "tables" / t).is_file(), f"tables/{t} written")
    p1 = tests[tests.MODEL.astype(str).str.startswith("P1_") & tests.TEST.eq("ROH")]
    p2 = tests[tests.MODEL.astype(str).str.startswith("P2_")]
    n_regions = tests.REGION_ID.nunique()
    check(len(p1) == n_regions and (p1.CONVERGED == 1).all(), f"Model 1 (ROH + COV + APOE4) converged for all {n_regions} region(s)")
    check(len(p2) == 2 * n_regions and ((p2.CONVERGED == 1) | p2.STATUS.astype(str).str.startswith("insufficient")).all(),
          "Model 2 (ROH x APOE4) converged or is reported with a pre-specified not-estimable status")
    ap_audit = json.loads((out / "apoe_audit.json").read_text())
    check(ap_audit.get("dosage_available_N", 0) > 0 and ap_audit.get("mapping_encoding") == "epsilon4_dosage",
          f"APOE decoded as epsilon4 dosage (N={ap_audit.get('dosage_available_N')})")
    if a.kinship:
        k = tests[tests.MODEL.astype(str).str.contains("GLMM")]
        check(len(k) > 0 and (k.CONVERGED == 1).any(), "kinship GLMM models fitted")
    if a.apoe_la:
        check(tests.MODEL.astype(str).str.startswith("E1_").any(), "APOE-locus local-ancestry (E1) models planned")
    figs = list((out / "figures").glob("*"))
    for ext in ("pdf", "svg", "png", "tiff"):
        check(any(f.suffix == "." + ext for f in figs), f"figures written as .{ext}")
    check(any(f.name.startswith("FigS2_all_tests_overview") for f in figs), "all-tests overview figure written")
if failures:
    raise SystemExit(f"{len(failures)} end-to-end check(s) failed")
print("All end-to-end checks passed")
