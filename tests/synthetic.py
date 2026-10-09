"""Synthetic completed-TRUROH run used by the tests and the demo report.

Simulated (NOT real data): a Caribbean-Hispanic-like admixed cohort with global ancestry (EUR/AFR/NAM),
one high-confidence ROH region, local ancestry at that region (carriers are homozygous for one ancestry),
APOE_reported coded as bare 0/1/2 epsilon4 dosage (the v1 failure mode), and an AD outcome with
APOE and ancestry-specific APOE effects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trurohassoc.common import write_json, write_tsv  # noqa: E402


def make_run(directory, n=2278, n_carriers=56, seed=1248, apoe_code=lambda d: str(int(d)), region_hc=1,
             roh_beta=-0.35, roh_apoe_beta=0.0, family_fraction=0.45, family_sd=0.9):
    directory = Path(directory)
    run = directory / "completed run"
    (run / "inputs").mkdir(parents=True)
    (run / "results").mkdir()
    rng = np.random.default_rng(seed)
    ids = [f"{i:05d}" for i in range(n - 1)] + ["NA"]
    anc = ["EUR", "AFR", "NAM"]
    # families of 2-4 siblings (about family_fraction of the cohort); the rest unrelated
    fam = np.full(n, -1)
    i, f = 0, 0
    while i < int(n * family_fraction):
        size = int(rng.integers(2, 5))
        fam[i:i + size] = f
        i, f = i + size, f + 1
    singles = fam < 0
    fam[singles] = np.arange(f, f + singles.sum())
    q_fam = rng.dirichlet([6, 3, 1.2], fam.max() + 1)
    q = np.clip(q_fam[fam] + rng.normal(0, 0.02, (n, 3)), 1e-3, None)
    q = q / q.sum(axis=1, keepdims=True)
    familial = rng.normal(0, family_sd, fam.max() + 1)[fam]
    # local ancestry at the ROH locus: two haplotypes, ancestry ~ global ancestry
    hap = np.stack([np.array([rng.choice(3, p=row) for row in q]) for _ in range(2)], axis=1)
    # ROH carriers: homozygous (IBD) haplotypes, NAM-enriched (bottleneck) -> both haplotypes one ancestry
    weight = q[:, 2] * 4 + q[:, 0] * 0.6 + q[:, 1] * 0.6
    roh = np.zeros(n)
    while roh.sum() < n_carriers:
        i = int(rng.integers(n))
        if roh[i]:
            continue
        if rng.random() < weight[i] / weight.max() * 0.5:
            roh[i] = 1
            for j in np.flatnonzero(fam == fam[i]):      # siblings share the IBD segment half the time
                if j != i and roh.sum() < n_carriers and rng.random() < 0.5:
                    roh[j] = 1
    carriers = np.flatnonzero(roh == 1)
    for i in carriers:
        a = rng.choice(3, p=np.array([q[i, 0] * .6, q[i, 1] * .6, q[i, 2] * 4]) / (q[i, 0] * .6 + q[i, 1] * .6 + q[i, 2] * 4))
        hap[i] = [a, a]
    la = np.stack([(hap == k).sum(axis=1) / 2 for k in range(3)], axis=1)
    e4_freq = 0.10 + 0.08 * q[:, 1]
    dosage = rng.binomial(2, e4_freq)
    age = rng.normal(74, 8, n).clip(55, 98)
    sex = rng.binomial(1, 0.62, n)
    pc = np.column_stack([q[:, 1] - q[:, 1].mean() + rng.normal(0, .02, n), q[:, 2] - q[:, 2].mean() + rng.normal(0, .02, n),
                          rng.normal(0, .02, (n, 3))])
    apoe_effect = 0.95 - 0.45 * (q[:, 1] * 10 / 3)        # weaker e4 effect with more African ancestry
    eta = (-0.75 + 0.45 * (age - 74) / 10 + 0.15 * sex + apoe_effect * dosage + roh_beta * roh
           + roh_apoe_beta * roh * dosage + 0.25 * (q[:, 1] - .3) * 10 / 3 + familial)
    ad = rng.binomial(1, expit(eta)).astype(float)
    ad[rng.choice(n, 12, replace=False)] = np.nan
    pheno = pd.DataFrame({"FID": [f"FAM{k:05d}" for k in fam], "IID": ids, "AD": ad, "AGE": age, "SEX": sex, "APOE4": np.nan})
    for j in range(5):
        pheno[f"PC{j + 1}"] = pc[:, j]
    write_tsv(run / "inputs/phenotypes_prepared.tsv", pheno)
    raw = pd.DataFrame({"FID": pheno.FID, "IID": pheno.IID, "Sex": sex, "Age": age.round(1),
                        "APOE_reported": [apoe_code(d) for d in dosage], "AD": ad})
    for j in range(5):
        raw[f"PC{j + 1}"] = pc[:, j]
    raw.loc[rng.choice(n, 40, replace=False), "APOE_reported"] = "-9"
    write_tsv(directory / "source phenotype.tsv", raw)
    cfg = {"project": "CH", "cohort_label": "Synthetic CH-like cohort", "output_root": str(run),
           "phenotype_file": str(directory / "source phenotype.tsv"),
           "columns": {"fid": "FID", "iid": "IID", "age": "Age", "sex": "Sex", "apoe_genotype": "APOE_reported"},
           "missing_values": ["", "NA", "-9", "."],
           "association": {"covariates": ["AGE", "SEX", "PC1", "PC2", "PC3", "PC4", "PC5"],
                           "ancestries": ["NAM", "EUR", "AFR"], "global_reference": "EUR"}}
    write_json(run / "config.resolved.json", cfg)
    write_json(directory / "ch.json", cfg)
    write_json(run / "run_manifest.json", {"project": "CH", "cohort": "Synthetic CH-like cohort", "technical_N": n,
                                           "run_signature": "fixture-signature", "locked": {"GRID_ID": 1433}})
    write_json(run / "final_status.json", {"state": "completed", "run_signature": "fixture-signature"})
    region = {"REGION_ID": "bcftools.no.chr6.R000648", "CHR": 6, "CONS_START": 57315049, "CONS_END": 58402135,
              "N_CARRIERS": n_carriers, "HIGH_CONFIDENCE": region_hc, "GENES": "GUSBP4;PRIM2"}
    # background (low-confidence) candidate regions -> genome-wide F_ROH; more autozygosity with NAM ancestry
    inbred = rng.gamma(1.5, 0.006 + 0.02 * q[:, 2])
    regions, mem = [region], [pd.DataFrame({"REGION_ID": region["REGION_ID"], "IID": np.array(ids)[roh == 1]})]
    for r in range(150):
        chrom = int(rng.choice([c for c in range(1, 23) if c != 6]))
        start = int(rng.integers(1_000_000, 100_000_000))
        length = int(rng.integers(1_000_000, 6_000_000))
        carry = rng.random(n) < np.clip(inbred * 4, 0, 0.9)
        rid = f"bcftools.no.chr{chrom}.B{r:05d}"
        regions.append({"REGION_ID": rid, "CHR": chrom, "CONS_START": start, "CONS_END": start + length,
                        "N_CARRIERS": int(carry.sum()), "HIGH_CONFIDENCE": 0, "GENES": ""})
        mem.append(pd.DataFrame({"REGION_ID": rid, "IID": np.array(ids)[carry]}))
    write_tsv(run / "results/regions_scored.tsv", pd.DataFrame(regions))
    write_tsv(run / "results/members.tsv.gz", pd.concat(mem, ignore_index=True))
    # KING-style kinship table: siblings ~0.25, plus unrelated pairs near 0
    pairs = []
    for k in np.unique(fam):
        idx = np.flatnonzero(fam == k)
        for a_ in range(len(idx)):
            for b_ in range(a_ + 1, len(idx)):
                pairs.append((ids[idx[a_]], ids[idx[b_]], 0.25 + rng.normal(0, 0.02)))
    for _ in range(3000):
        a_, b_ = rng.choice(n, 2, replace=False)
        if fam[a_] != fam[b_]:
            pairs.append((ids[a_], ids[b_], rng.normal(0, 0.01)))
    king = pd.DataFrame(pairs, columns=["IID1", "IID2", "KINSHIP"])
    king.columns = ["#IID1", "IID2", "KINSHIP"]
    king.to_csv(directory / "cohort.kin0", sep="\t", index=False)
    gq = pd.DataFrame(q, columns=["Q_European", "Q_African", "Q_American"])
    gq.insert(0, "IID", ids)
    write_tsv(run / "results/global_ancestry.tsv", gq)
    lf = pd.DataFrame(la, columns=["LA_EUR", "LA_AFR", "LA_NAM"])
    lf.insert(0, "IID", ids)
    lf.insert(0, "REGION_ID", region["REGION_ID"])
    lf["ROH"] = roh.astype(int)
    write_tsv(run / "results/local_ancestry_for_association.tsv.gz", lf)
    prior = pd.DataFrame([
        {"REGION_ID": region["REGION_ID"], "MODEL": "M0", "TESTED_TERM": "ROH", "STATUS": "converged", "CONVERGED": 1, "P_VALUE": .249157, "OR": .716458, "CI_LOW": .399548, "CI_HIGH": 1.284731, "N": n - 12},
        {"REGION_ID": region["REGION_ID"], "MODEL": "M1", "TESTED_TERM": "ROH", "STATUS": "converged", "CONVERGED": 1, "P_VALUE": .237688, "OR": .703296, "CI_LOW": .385603, "CI_HIGH": 1.282732, "N": n - 12},
        {"REGION_ID": region["REGION_ID"], "MODEL": "ROHxE", "TESTED_TERM": "ROHxAPOE4", "STATUS": "no_complete_samples", "CONVERGED": 0, "P_VALUE": np.nan}])
    write_tsv(run / "results/associations.tsv", prior)
    # APOE-locus local ancestry (independent draw around global ancestry)
    hap19 = np.stack([np.array([rng.choice(3, p=row) for row in q]) for _ in range(2)], axis=1)
    la19 = pd.DataFrame(np.stack([(hap19 == k).sum(axis=1) / 2 for k in range(3)], axis=1), columns=["LA_EUR", "LA_AFR", "LA_NAM"])
    la19.insert(0, "IID", ids)
    write_tsv(directory / "apoe_locus_la.tsv", la19)
    write_json(directory / "map.json", {"encoding": "epsilon4_dosage", "mapping": {"0": 0, "1": 1, "2": 2}})
    return run, cfg, pheno, region


if __name__ == "__main__":
    make_run(sys.argv[1])
