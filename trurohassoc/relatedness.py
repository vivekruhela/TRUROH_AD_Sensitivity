"""Relatedness (kinship) and genome-wide autozygosity (F_ROH).

Kinship enters the analysis in three ways - never as a single 'kinship covariate', because kinship is a
property of pairs, not of individuals:
  1. GLMM random effect with covariance tau * GRM (GRM = 2 x kinship)          -> glmm.py
  2. cluster-robust (sandwich) variances with relative clusters as units       -> glmm.cluster_robust_cov
  3. a maximal unrelated subset (no pair closer than 2nd degree)
The individual-level quantity that *is* a covariate is the inbreeding coefficient (here genome-wide F_ROH,
leave-focal-region-out), i.e. kinship between the person's own parents.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import read_tsv, sha256

DEGREE_CUTOFFS = [("duplicate/MZ", 0.354), ("1st degree", 0.177), ("2nd degree", 0.0884), ("3rd degree", 0.0442)]
RELATED_THRESHOLD = 0.0884          # kinship coefficient; 2nd degree or closer
AUTOSOME_BP = 2_875_001_522         # GRCh38 autosomes


class Relatedness:
    def __init__(self, ids, kinship_pairs, grm=None, source="", clusters_from="kinship", fids=None):
        self.ids = list(ids)
        self.index = {iid: i for i, iid in enumerate(self.ids)}
        self.pairs = kinship_pairs            # DataFrame IID1, IID2, KINSHIP (cohort members only)
        self.grm = grm                        # dense N x N GRM or None
        self.source = source
        self.clusters_from = clusters_from
        related = kinship_pairs[kinship_pairs.KINSHIP >= RELATED_THRESHOLD] if len(kinship_pairs) else kinship_pairs
        parent = {i: i for i in self.ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        if clusters_from == "FID" and fids is not None:
            self.cluster = dict(zip(self.ids, fids))
        else:
            for a, b in zip(related.IID1, related.IID2):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
            self.cluster = {i: find(i) for i in self.ids}
        self.related_edges = {}
        if clusters_from == "FID" and fids is not None:
            groups = pd.Series(self.ids).groupby(pd.Series(fids)).apply(list)
            for members in groups:
                for a in members:
                    self.related_edges[a] = set(members) - {a}
        else:
            for a, b in zip(related.IID1, related.IID2):
                self.related_edges.setdefault(a, set()).add(b)
                self.related_edges.setdefault(b, set()).add(a)

    def matrix_for(self, iids):
        if self.grm is None:
            raise ValueError("No kinship matrix (supply --kinship)")
        idx = np.array([self.index[i] for i in iids])
        return self.grm[np.ix_(idx, idx)]

    def unrelated(self, iids, prefer_keep):
        """Greedy maximal unrelated set: repeatedly drop the person with most relatives; ties drop
        non-preferred (e.g. non-carrier / incomplete covariates) first, then by ID. Never uses the outcome."""
        keep = set(iids)
        edges = {i: set(self.related_edges.get(i, set())) & keep for i in keep}
        while True:
            worst = [i for i in keep if edges[i]]
            if not worst:
                return keep
            drop = max(worst, key=lambda i: (len(edges[i]), not prefer_keep.get(i, False), i))
            for j in edges[drop]:
                edges[j].discard(drop)
            keep.discard(drop)
            edges.pop(drop)

    def summary(self, iids=None):
        ids = set(self.ids if iids is None else iids)
        p = self.pairs[self.pairs.IID1.isin(ids) & self.pairs.IID2.isin(ids)] if len(self.pairs) else self.pairs
        out = {"source": self.source, "clusters_from": self.clusters_from, "N": len(ids)}
        if self.clusters_from == "kinship":
            upper = np.inf
            for name, cut in DEGREE_CUTOFFS:
                out[f"pairs_{name}"] = int(((p.KINSHIP >= cut) & (p.KINSHIP < upper)).sum()) if len(p) else 0
                upper = cut
        with_rel = [i for i in ids if self.related_edges.get(i, set()) & ids]
        out["individuals_with_2nd_degree_or_closer_relative"] = len(with_rel)
        sizes = pd.Series([self.cluster[i] for i in ids]).value_counts()
        out["clusters"] = int(len(sizes))
        out["largest_cluster"] = int(sizes.max()) if len(sizes) else 0
        return out


def _read_king(path, cohort):
    t = pd.read_csv(path, sep=r"\s+", dtype=str, keep_default_na=False)
    t.columns = [c.lstrip("#") for c in t.columns]
    lower = {c.lower(): c for c in t.columns}
    id1 = lower.get("iid1") or lower.get("id1")
    id2 = lower.get("iid2") or lower.get("id2")
    kin = lower.get("kinship")
    if not (id1 and id2 and kin):
        raise ValueError(f"{path}: need IID1/ID1, IID2/ID2 and KINSHIP columns (KING .kin0 or plink2 --make-king-table)")
    t = pd.DataFrame({"IID1": t[id1], "IID2": t[id2], "KINSHIP": pd.to_numeric(t[kin], errors="coerce")})
    t = t[t.IID1.isin(cohort) & t.IID2.isin(cohort) & (t.IID1 != t.IID2) & t.KINSHIP.notna()]
    return t


def _read_gcta(prefix, cohort):
    ids = pd.read_csv(str(prefix) + ".grm.id", sep=r"\s+", header=None, dtype=str, keep_default_na=False)
    n = len(ids)
    vals = np.fromfile(str(prefix) + ".grm.bin", dtype=np.float32)
    if len(vals) != n * (n + 1) // 2:
        raise ValueError("GCTA .grm.bin size does not match .grm.id")
    full = np.zeros((n, n))
    full[np.tril_indices(n)] = vals
    full = full + np.tril(full, -1).T
    return list(ids[1]), full


def load_relatedness(kinship, pheno, output_dir=None):
    """kinship: KING table path, GCTA GRM prefix (….grm.bin/.grm.id), or None (FID clusters only)."""
    cohort = list(pheno.IID)
    cset = set(cohort)
    if kinship is None:
        rel = Relatedness(cohort, pd.DataFrame(columns=["IID1", "IID2", "KINSHIP"]), None,
                          source="FID (no kinship file)", clusters_from="FID", fids=list(pheno.FID))
        return rel, dict(rel.summary(), note="No kinship file: GLMM unavailable; robust SEs and unrelated subset use FID families")
    path = Path(kinship)
    prefix = Path(str(path).removesuffix(".grm.bin").removesuffix(".grm.id"))
    if Path(str(prefix) + ".grm.bin").is_file():
        ids, full = _read_gcta(prefix, cset)
        pos = {i: k for k, i in enumerate(ids)}
        missing = [i for i in cohort if i not in pos]
        if missing:
            raise ValueError(f"GRM lacks {len(missing)} cohort IIDs (e.g. {missing[:3]})")
        idx = np.array([pos[i] for i in cohort])
        grm = full[np.ix_(idx, idx)]
        iu = np.triu_indices(len(cohort), 1)
        keep = grm[iu] / 2 >= 0.0221
        pairs = pd.DataFrame({"IID1": np.array(cohort)[iu[0][keep]], "IID2": np.array(cohort)[iu[1][keep]],
                              "KINSHIP": grm[iu][keep] / 2})
        source, digest = f"GCTA GRM {prefix}", sha256(str(prefix) + ".grm.bin")
    else:
        pairs = _read_king(path, cset)
        grm = np.eye(len(cohort))
        pos = {i: k for k, i in enumerate(cohort)}
        sparse = pairs[pairs.KINSHIP >= 0.0221]     # sparse GRM: 4th degree or closer, negatives -> 0
        a = sparse.IID1.map(pos).to_numpy()
        b = sparse.IID2.map(pos).to_numpy()
        grm[a, b] = grm[b, a] = 2 * sparse.KINSHIP.to_numpy()
        source, digest = f"KING table {path}", sha256(path)
    lam, vec = np.linalg.eigh(grm)
    clipped = int((lam < -1e-8).sum())
    if clipped:
        grm = (vec * np.clip(lam, 0, None)) @ vec.T
    rel = Relatedness(cohort, pairs.reset_index(drop=True), grm, source=source)
    return rel, dict(rel.summary(), sha256=digest, negative_eigenvalues_clipped=clipped)


def read_all_segments(root, scored):
    """All candidate-region memberships with coordinates (IID, CHR, START, END) for genome-wide F_ROH."""
    coords = scored.set_index("REGION_ID")[["CHR", "CONS_START", "CONS_END"]]
    pieces = []
    for chunk in pd.read_csv(Path(root) / "results/members.tsv.gz", sep="\t", dtype=str, keep_default_na=False,
                             usecols=["REGION_ID", "IID"], chunksize=500000):
        chunk = chunk.join(coords, on="REGION_ID", how="inner")
        pieces.append(chunk[["IID", "CHR", "CONS_START", "CONS_END"]])
    seg = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["IID", "CHR", "CONS_START", "CONS_END"])
    seg.columns = ["IID", "CHR", "START", "END"]
    for c in ["CHR", "START", "END"]:
        seg[c] = pd.to_numeric(seg[c], errors="coerce")
    return seg.dropna().sort_values(["IID", "CHR", "START"]).reset_index(drop=True)


def froh_excluding(segments, iids, chrom=None, start=None, end=None):
    """Percent of the autosomes in candidate ROH regions (interval union), excluding regions that overlap
    the focal region so the covariate is not mechanically the exposure."""
    s = segments
    if chrom is not None:
        s = s[~((s.CHR == chrom) & (s.END > start) & (s.START < end))]
    if s.empty:
        return pd.Series(0.0, index=list(iids))
    prev = s.groupby(["IID", "CHR"]).END.cummax().groupby([s.IID, s.CHR]).shift(1)
    length = (s.END - np.maximum(s.START, prev.fillna(s.START))).clip(lower=0)
    total = length.groupby(s.IID).sum()
    return (100.0 * total / AUTOSOME_BP).reindex(list(iids), fill_value=0.0)


def read_inbreeding(path, iids):
    t = read_tsv(path)
    col = next((c for c in t.columns if c.upper() in ("F", "F_ROH", "FROH", "F_INBREEDING")), None)
    if "IID" not in t or col is None:
        raise ValueError(f"{path}: need IID and one of F / F_ROH / FROH columns (proportion 0-1)")
    values = pd.to_numeric(t.set_index("IID")[col], errors="coerce")
    return (100.0 * values).reindex(list(iids))
