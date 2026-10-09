#!/usr/bin/env python3
"""Pre-submission check used by submit.sh: parse the analysis arguments exactly as the job will, locate
the completed TRU-ROH run and the output directory, and confirm every named file exists. Nothing is written.

    python3 tools/check_args.py [--allow-pending-run] <roh_assoc.py arguments>
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trurohassoc.cli import locate, parse  # noqa: E402
from trurohassoc.common import read_json  # noqa: E402
from trurohassoc.data import REQUIRED  # noqa: E402


def main(argv):
    pending = "--allow-pending-run" in argv
    argv = [a for a in argv if a != "--allow-pending-run"]
    args = parse(argv)
    root, output = locate(args)
    problems, notes = [], []
    missing = [p for p in REQUIRED if not (root / p).is_file()]
    if missing:
        (notes if pending else problems).append(f"completed TRU-ROH run not found at {root} (missing {', '.join(missing)})")
    else:
        state = read_json(root / "final_status.json").get("state")
        if state != "completed":
            (notes if pending else problems).append(f"TRU-ROH run {root} has state '{state}', not 'completed'")
        else:
            hc = sum(1 for _ in open(root / "results/regions_scored.tsv")) - 1
            notes.append(f"completed TRU-ROH run: {root} ({hc} scored candidate regions)")
    if output.exists():
        problems.append(f"output directory already exists: {output}")
    for key in ("apoe_map", "apoe_source", "apoe_locus_la", "inbreeding"):
        value = getattr(args, key, None)
        if value is not None and not Path(value).is_file():
            problems.append(f"--{key.replace('_', '-')} file not found: {value}")
    if args.kinship is not None:
        k = Path(args.kinship)
        if not (k.is_file() or Path(str(k) + ".grm.bin").is_file()):
            problems.append(f"--kinship not found (KING table or GCTA prefix): {k}")
    print(f"Output (created by the job): {output if not args.out_parent else str(args.out_parent) + '/<name>_<UTC time>'}")
    for n in notes:
        print("  note: " + n)
    for p in problems:
        print("  ERROR: " + p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception as exc:  # argument/locate errors
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
