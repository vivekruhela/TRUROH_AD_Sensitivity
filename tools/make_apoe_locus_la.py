#!/usr/bin/env python3
"""Build apoe_locus_LA.tsv (local ancestry at APOE) from an RFMix v2 chr19 .msp.tsv file.

Output (tab-separated, one row per individual):
    IID     LA_AFR  LA_EUR  LA_NAM
    00001   0.0     0.5     0.5
Each LA_<pop> is the fraction of the individual's two haplotypes assigned to <pop> at APOE
(0, 0.5 or 1; values in between only when several RFMix windows overlap APOE and disagree).
The analysis converts it to 0/1/2 haplotypes internally.

Example:
    python3 tools/make_apoe_locus_la.py --msp /path/to/rfmix/chr19.msp.tsv --out apoe_locus_LA.tsv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# APOE gene, GRCh38 (rs429358 = 44,908,684; rs7412 = 44,908,822). Use --start/--end for GRCh37 (45,409,011-45,412,650).
APOE_GRCH38 = (44_905_754, 44_909_393)
ALIASES = {"AMR": "NAM", "American": "NAM", "Native_American": "NAM", "European": "EUR", "African": "AFR"}


def read_msp(path):
    with open(path) as handle:
        first = handle.readline()
    codes = dict(re.findall(r"([^\s=:]+)=(\d+)", first.split(":", 1)[-1]))
    if not codes:
        raise ValueError("First line must be the RFMix '#Subpopulation order/codes: POP=0 ...' header")
    msp = pd.read_csv(path, sep="\t", skiprows=1)
    msp.columns = [c.lstrip("#").strip() for c in msp.columns]
    return {int(v): ALIASES.get(k, k) for k, v in codes.items()}, msp


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--msp", required=True, type=Path, help="RFMix v2 .msp.tsv for chromosome 19")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--start", type=int, default=APOE_GRCH38[0], help="APOE start (default GRCh38)")
    p.add_argument("--end", type=int, default=APOE_GRCH38[1], help="APOE end (default GRCh38)")
    p.add_argument("--max-distance", type=int, default=500_000,
                   help="If no window overlaps APOE, use the nearest window within this many bp")
    p.add_argument("--id-map", type=Path, help="Optional TSV with columns VCF_ID and IID if VCF sample names differ from IIDs")
    a = p.parse_args(argv)

    codes, msp = read_msp(a.msp)
    chrom = msp["chm"].astype(str).str.replace("chr", "", regex=False)
    msp = msp[chrom == "19"].copy()
    if msp.empty:
        raise ValueError("No chromosome 19 windows in the .msp file")
    overlap = (np.minimum(msp.epos, a.end) - np.maximum(msp.spos, a.start)).clip(lower=0)
    if (overlap > 0).any():
        windows, weights = msp[overlap > 0], overlap[overlap > 0].to_numpy(float)
        note = f"{len(windows)} window(s) overlapping APOE"
    else:
        distance = np.where(msp.epos < a.start, a.start - msp.epos, msp.spos - a.end)
        nearest = int(np.argmin(distance))
        if distance[nearest] > a.max_distance:
            raise ValueError(f"Nearest RFMix window is {distance[nearest]:,} bp from APOE (> --max-distance)")
        windows, weights = msp.iloc[[nearest]], np.ones(1)
        note = f"nearest window, {distance[nearest]:,} bp from APOE"
    weights = weights / weights.sum()

    hap_cols = [c for c in msp.columns if re.search(r"\.[01]$", c)]
    samples = list(dict.fromkeys(c[:-2] for c in hap_cols))
    pops = [codes[k] for k in sorted(codes)]
    rows = {}
    for s in samples:
        calls = windows[[s + ".0", s + ".1"]].to_numpy(int)          # windows x 2 haplotypes
        rows[s] = [float(np.sum(weights[:, None] * (calls == k)) / 2) for k in sorted(codes)]
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["LA_" + x for x in pops])
    out.index.name = "IID"
    out = out.reset_index()
    if a.id_map:
        idmap = pd.read_csv(a.id_map, sep="\t", dtype=str, keep_default_na=False)
        out = out.merge(idmap, left_on="IID", right_on="VCF_ID", how="inner", suffixes=("_vcf", ""))
        out = out[["IID"] + ["LA_" + x for x in pops]]
    if out.IID.duplicated().any():
        raise ValueError("Duplicate IIDs after mapping")
    out.to_csv(a.out, sep="\t", index=False, float_format="%.4g")
    print(f"Wrote {len(out)} individuals, ancestries {pops}, using {note} "
          f"({windows.spos.min():,}-{windows.epos.max():,}).", file=sys.stderr)


if __name__ == "__main__":
    main()
