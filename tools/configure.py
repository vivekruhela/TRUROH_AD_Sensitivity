#!/usr/bin/env python3
"""Interactive configuration wizard for the TRU-ROH AD association / sensitivity pipeline.

It asks where the completed TRU-ROH run is, how APOE is recorded, which optional relatedness and
APOE-locus files to use, where to write results and how to submit to SLURM, while inspecting what you
name. It then writes a profile (``profiles/<name>.json``) that ``run.sh`` / ``submit.sh`` accept with
``--profile <name>``, and ``config/site.env`` with the scheduler resources and module/conda commands.

  bash configure.sh                              # interactive
  bash configure.sh --answers answers.json       # non-interactive (keys = the names in [brackets])

When standard input is not a terminal, answers are read line by line (an empty line accepts the default).
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trurohassoc.data import REQUIRED  # noqa: E402

ANSWERS: dict = {}
INTERACTIVE = sys.stdin.isatty()


def ask(key, question, default="", validate=None, help_text=""):
    default = "" if default is None else str(default)
    while True:
        if key in ANSWERS:
            value = str(ANSWERS[key]) if ANSWERS[key] is not None else ""
            print(f"[{key}] {question}: {value}")
        else:
            if help_text and INTERACTIVE:
                print("    " + help_text)
            prompt = f"[{key}] {question}" + (f" [{default}]" if default else "") + ": "
            if INTERACTIVE:
                value = input(prompt).strip()
            else:
                print(prompt, end="")
                line = sys.stdin.readline()
                value = line.strip()
                print(value)
            value = value or default
        if validate is None:
            return value
        try:
            out = validate(value)
            return value if out is None else out
        except ValueError as exc:
            print(f"    ! {exc}")
            if key in ANSWERS or not INTERACTIVE:
                raise SystemExit(f"Invalid answer for {key}: {exc}")


def optional_file(label, grm_ok=False):
    def check(v):
        if not v:
            return ""
        p = Path(v).expanduser().resolve()
        if p.is_file() or (grm_ok and Path(str(p) + ".grm.bin").is_file()):
            return str(p)
        raise ValueError(f"{label} not found: {p}")
    return check


def yes_no(v):
    if v.lower() in ("y", "yes", "1", "true"):
        return True
    if v.lower() in ("n", "no", "0", "false", ""):
        return False
    raise ValueError("answer y or n")


def positive_int(v):
    if not str(v).isdigit() or int(v) < 1:
        raise ValueError("enter a positive integer")
    return int(v)


def locate_run(v):
    p = Path(v).expanduser().resolve()
    if p.is_file() and p.suffix == ".json":
        cfg = json.loads(p.read_text())
        root = Path(cfg.get("output_root", "")).expanduser()
        root = root if root.is_absolute() else (p.parent / root).resolve()
        return {"config": str(p), "root": root}
    if p.is_dir():
        return {"run_dir": str(p), "root": p}
    raise ValueError(f"neither a cohort JSON nor a directory: {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", type=Path, help="JSON of pre-supplied answers (non-interactive)")
    ap.add_argument("--profile-dir", type=Path, default=ROOT / "profiles")
    ap.add_argument("--site-env", type=Path, default=ROOT / "config" / "site.env")
    a = ap.parse_args()
    if a.answers:
        ANSWERS.update(json.loads(a.answers.read_text()))

    print("TRU-ROH AD association — configuration\n" + "=" * 40)
    print("\n1. Completed TRU-ROH run")
    run = ask("run", "TRU-ROH cohort JSON (TRUROH_Ancestry config) or completed output directory", "", locate_run,
              "The run must have finished: <output_root>/final_status.json state 'completed'.")
    root = run["root"]
    missing = [p for p in REQUIRED if not (root / p).is_file()]
    if missing:
        print(f"    ! {root} is not (yet) a completed run: missing {', '.join(missing)}.")
        print("      The profile is still written; submit with TRUROH_AD_DEPENDENCY=afterok:<finalizer job id>.")
        cfg = json.loads(Path(run["config"]).read_text()) if "config" in run else {}
    else:
        final = json.loads((root / "final_status.json").read_text())
        cfg = json.loads((root / "config.resolved.json").read_text())
        hc_file = root / "results/high_confidence_ROH.tsv"
        n_hc = len(pd.read_csv(hc_file, sep="\t")) if hc_file.is_file() else "unknown"
        print(f"    run state: {final.get('state')}; project {cfg.get('project')}; high-confidence ROH regions: {n_hc}")
        if final.get("state") != "completed":
            print("    ! the run is not completed; the analysis will refuse to start until it is.")
    project = cfg.get("project", "cohort")

    print("\n2. APOE")
    saved_source = cfg.get("phenotype_file", "")
    saved_col = (cfg.get("columns") or {}).get("apoe_genotype", "")
    source = ask("apoe_source", "Table holding APOE (TAB-separated; blank = the TRU-ROH phenotype file)", "",
                 optional_file("APOE source"))
    table = source or saved_source
    header, values = [], []
    if table and Path(table).is_file():
        df = pd.read_csv(table, sep="\t", dtype=str, nrows=200000)
        header = list(df.columns)
        print("    columns: " + ", ".join(header))
    default_col = saved_col if saved_col in header or not header else next((c for c in header if "apoe" in c.lower()), "")
    column = ask("apoe_column", "APOE column", default_col)
    if header and column and column not in header:
        raise SystemExit(f"APOE column '{column}' not in {table}")
    if header and column:
        values = sorted(df[column].dropna().astype(str).str.strip().unique().tolist())
        print(f"    observed values: {', '.join(values[:15])}{' …' if len(values) > 15 else ''}")
    numeric_codes = bool(values) and set(values) <= {"0", "1", "2"}
    map_default = "@pkg/apoe_dosage_map.json" if numeric_codes else ""
    apoe_map = ask("apoe_map", "APOE code map (bundled @pkg/apoe_dosage_map.json = 0/1/2 epsilon4 alleles; blank if genotype strings like e3/e4)",
                   map_default, help_text="Confirm the coding with the cohort's data dictionary.")
    if apoe_map and not apoe_map.startswith("@pkg/"):
        apoe_map = optional_file("APOE map")(apoe_map)
    overrides_needed = bool(source) or (column and column != saved_col)
    if overrides_needed and not apoe_map:
        raise SystemExit("Using an APOE source/column other than the saved one requires an explicit APOE map.")
    id_cols = {}
    if source:
        id_cols["apoe_fid_column"] = ask("apoe_fid_column", "FID column in the APOE table", "FID" if "FID" in header else "")
        id_cols["apoe_iid_column"] = ask("apoe_iid_column", "IID column in the APOE table", "IID" if "IID" in header else "")

    print("\n3. Relatedness and optional inputs")
    kinship = ask("kinship", "Kinship: KING .kin0 table or GCTA GRM prefix (blank = none; recommended if relatives exist)", "",
                  optional_file("kinship", grm_ok=True))
    primary = "firth"
    if kinship:
        primary = ask("primary", "Primary model family: firth or glmm (decide before seeing results)", "firth")
        if primary not in ("firth", "glmm"):
            raise SystemExit("primary must be firth or glmm")
    inbreeding = ask("inbreeding", "Own inbreeding estimate TSV (IID, F) (blank = F_ROH from TRU-ROH regions)", "",
                     optional_file("inbreeding file"))
    apoe_la = ask("apoe_locus_la", "Local ancestry at APOE TSV from tools/make_apoe_locus_la.py (blank = skip E1 models)", "",
                  optional_file("APOE-locus local ancestry"))
    min_carriers = ask("min_carriers", "Minimum ROH carriers and non-carriers per model", 3, positive_int)

    print("\n4. Output")
    out_parent = ask("out_parent", "Parent directory for results (blank = next to the TRU-ROH run)", "")
    if out_parent:
        out_parent = str(Path(out_parent).expanduser().resolve())
    out_name = ask("out_name", "Result folder name (a UTC time stamp is appended)", f"{project}_ROH_APOE_ancestry")
    name = ask("profile_name", "Profile name", project.lower())

    print("\n5. SLURM and software")
    site = {
        "TRUROH_AD_PARTITION": ask("slurm.partition", "Partition (blank = cluster default)", ""),
        "TRUROH_AD_ACCOUNT": ask("slurm.account", "Account (blank = none)", ""),
        "TRUROH_AD_CONSTRAINT": ask("slurm.constraint", "Node constraint (blank = none)", ""),
        "TRUROH_AD_CPUS": ask("slurm.cpus", "CPUs", 4, positive_int),
        "TRUROH_AD_MEMORY": ask("slurm.memory", "Memory", "16G"),
        "TRUROH_AD_TIME": ask("slurm.time", "Time limit", "1-00:00:00"),
    }
    modules = ask("site.modules", "Cluster modules to load in the job (blank = none)", "")
    conda = ask("site.conda", "Conda environment to activate in the job (blank = none)", "")
    conda_sh = ask("site.conda_sh", "Path to conda.sh", "") if conda else ""
    python_bin = ask("site.python", "Python executable", "python3")

    profile = {"description": f"{cfg.get('cohort_label', project)} (written by configure.sh)"}
    profile.update({k: v for k, v in run.items() if k in ("config", "run_dir")})
    if apoe_map:
        profile["apoe_map"] = apoe_map
    if source:
        profile["apoe_source"] = source
    if column and (column != saved_col or source):
        profile["apoe_column"] = column
    profile.update({k: v for k, v in id_cols.items() if v})
    for key, value in [("kinship", kinship), ("inbreeding", inbreeding), ("apoe_locus_la", apoe_la), ("out_parent", out_parent)]:
        if value:
            profile[key] = value
    if primary != "firth":
        profile["primary"] = primary
    profile["min_carriers"] = int(min_carriers)
    profile["out_name"] = out_name

    a.profile_dir.mkdir(parents=True, exist_ok=True)
    path = a.profile_dir / f"{name}.json"
    path.write_text(json.dumps(profile, indent=2) + "\n")

    lines = ["# Written by configure.sh — sourced by submit.sh and the SLURM job.", ""]
    lines += [f"export {k}={shlex.quote(str(v))}" for k, v in site.items()]
    lines.append(f"export PYTHON_BIN={shlex.quote(python_bin)}")
    if modules:
        lines += ["type module >/dev/null 2>&1 || { [[ -r /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh; } || true",
                  "type module >/dev/null 2>&1 || { [[ -r /usr/share/lmod/lmod/init/bash ]] && source /usr/share/lmod/lmod/init/bash; } || true",
                  f"module load {modules}"]
    if conda:
        if conda_sh:
            lines.append(f"source {shlex.quote(conda_sh)}")
        lines.append(f"conda activate {shlex.quote(conda)}")
    a.site_env.parent.mkdir(parents=True, exist_ok=True)
    a.site_env.write_text("\n".join(lines) + "\n")

    # Validate exactly as the job will (no files written).
    import subprocess
    check = subprocess.run([sys.executable, str(ROOT / "tools/check_args.py"), "--allow-pending-run", "--profile", str(path)],
                           capture_output=True, text=True)
    print(check.stdout.rstrip())
    if check.returncode:
        print(check.stderr.rstrip())
        raise SystemExit(f"The profile {path} was written but did not validate; edit it or rerun configure.sh.")
    print(f"\nWrote {path}\nWrote {a.site_env}\n\nNext steps:")
    if a.site_env.resolve() != (ROOT / "config" / "site.env").resolve():
        print(f"  export TRUROH_AD_SITE_ENV={a.site_env.resolve()}")
    ref = name if path.parent.resolve() == (ROOT / "profiles").resolve() else str(path)
    print(f"  bash {ROOT}/run.sh --profile {ref} --audit-only      # seconds: check APOE decoding")
    print(f"  bash {ROOT}/submit.sh --dry-run --profile {ref}")
    print(f"  bash {ROOT}/submit.sh --profile {ref}")


if __name__ == "__main__":
    main()
