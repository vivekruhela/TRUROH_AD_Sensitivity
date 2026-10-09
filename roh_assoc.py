#!/usr/bin/env python3
"""Entry point: python3 roh_assoc.py --profile ch   (see README.md)."""
import os
import sys
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trurohassoc.cli import entry  # noqa: E402

if __name__ == "__main__":
    entry()
