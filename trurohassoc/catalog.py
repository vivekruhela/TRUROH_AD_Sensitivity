"""Fixed model catalog (written to analysis_manifest.json before any fit).

Design rules
------------
* Age, sex and PCs are adjustment covariates only - never interaction terms.
* Effect modifiers ("environment") are APOE e4 and ancestry (global Q, local LA at the ROH locus,
  ancestral origin of the ROH haplotype, optional local ancestry at APOE).
* Local-ancestry models are adjusted for global ancestry of the same component (primary) and repeated
  without it (sensitivity) - see README "Should global ancestry be included?".
* Every interaction model keeps all lower-order terms.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .common import unique

FAMILIES = {
    "primary": "Pre-specified primary hypotheses (Holm within family)",
    "apoe": "APOE coding / stratified sensitivity",
    "global_ancestry": "Global ancestry adjustment and ancestry-specific APOE / ROH effects",
    "local_ancestry": "Local ancestry at the ROH locus and ancestral origin of the ROH haplotype",
    "apoe_locus_ancestry": "Local ancestry at APOE (only when --apoe-locus-la is supplied)",
    "relatedness": "Kinship-aware versions of Models 1-2 (GLMM, cluster-robust, unrelated subset) and genome-wide F_ROH",
}

UNITS = {"ROH": "carrier vs non-carrier", "AGE_C": "per 10 years", "SEX": "SEX=1 vs 0",
         "APOE4": "per epsilon4 allele", "APOE_CARRIER": "epsilon4 carrier vs non-carrier",
         "APOE_1": "one vs zero epsilon4", "APOE_2": "two vs zero epsilon4"}


def unit_of(term):
    if term in UNITS:
        return UNITS[term]
    if ":" in term:
        return "ratio of odds ratios (interaction)"
    if term.startswith("Q10_"):
        return f"per 10% global {term[4:]} ancestry"
    if term.startswith("LAD_"):
        return f"per {term[4:]} haplotype at ROH locus"
    if term.startswith("LAA_"):
        return f"per {term[4:]} haplotype at APOE"
    if term.startswith("ROH_O_"):
        return f"{term[6:]}-origin ROH vs other-origin ROH"
    if term == "FROH_PCT":
        return "per 1% of autosomes in ROH (genome-wide, focal region excluded)"
    if term.startswith("PC"):
        return "per PC unit"
    return "per unit"


@dataclass
class Model:
    name: str
    label: str
    family: str
    terms: list
    tests: dict
    sample: str = "available"
    subset: dict | None = None
    discrete_support: list | None = None
    varying_support: list | None = None
    method: str = "firth"
    description: str = ""
    carrier_split: list | None = None
    test_family: dict = field(default_factory=dict)
    contrasts: list = field(default_factory=list)
    predict: dict | None = None


def _slope(group, base, slope, modifier, values):
    return [dict(group=group, modifier=modifier, value=float(v), weights={base: 1.0, slope: float(v)}) for v in values]


def model_catalog(core, ancestries, reference, apoe_locus=False, primary="firth"):
    core = list(core)
    nonref = [a for a in ancestries if a != reference]
    q_nonref = ["Q10_" + a for a in nonref]
    no_pc = [c for c in core if not c.startswith("PC")]
    models = []

    def add(name, label, family, covariates, tests=None, **kw):
        terms = unique(["ROH"] + list(covariates))
        models.append(Model(name, label, family, terms, tests or {"ROH": ["ROH"]}, **kw))

    # ---------------------------------------------------------------- primary (user models 1 and 2)
    add("P1_ROH_cov_APOE", "ROH + covariates + APOE e4 dosage", "primary", core + ["APOE4"],
        {"ROH": ["ROH"], "APOE4": ["APOE4"]}, test_family={"APOE4": "apoe"},
        description="Model 1: AD ~ ROH + age + sex + PC1-5 + APOE4 (original model with APOE restored)")
    add("P2_ROHxAPOE", "ROH x APOE e4 dosage", "primary", core + ["APOE4", "ROH:APOE4"],
        {"ROH:APOE4": ["ROH:APOE4"], "any_ROH": ["ROH", "ROH:APOE4"]},
        varying_support=["APOE4"],
        contrasts=_slope("ROH_by_APOE4", "ROH", "ROH:APOE4", "APOE4", [0, 1, 2]),
        predict={"ROH": [0, 1], "APOE4": [0, 1, 2]},
        description="Model 2: AD ~ ROH * APOE4 + age + sex + PC1-5")

    # ---------------------------------------------------------------- APOE coding / reproduction
    add("A0_unadjusted", "ROH only", "apoe", [])
    add("A1_cov_original", "ROH + covariates (original M1)", "apoe", core,
        description="Reproduces the original M1 (no APOE) on all samples")
    add("A1m_cov_APOEsample", "ROH + covariates, APOE-complete samples", "apoe", core, sample="apoe_dosage_complete",
        description="Same samples as P1: separates the APOE adjustment from the change of sample")
    add("A2_carrier", "ROH + covariates + e4 carrier", "apoe", core + ["APOE_CARRIER"],
        {"ROH": ["ROH"], "APOE_CARRIER": ["APOE_CARRIER"]})
    add("A3_genotypic", "ROH + covariates + e4 genotype (1, 2 copies)", "apoe", core + ["APOE_1", "APOE_2"],
        {"ROH": ["ROH"], "APOE_genotype": ["APOE_1", "APOE_2"]})
    add("A4_ROHxcarrier", "ROH x e4 carrier", "apoe", core + ["APOE_CARRIER", "ROH:APOE_CARRIER"],
        {"ROH:APOE_CARRIER": ["ROH:APOE_CARRIER"], "any_ROH": ["ROH", "ROH:APOE_CARRIER"]},
        discrete_support=["APOE_CARRIER"],
        contrasts=_slope("ROH_by_APOE_carrier", "ROH", "ROH:APOE_CARRIER", "APOE_CARRIER", [0, 1]))
    add("A5_ROHxgenotype", "ROH x e4 genotype", "apoe", core + ["APOE_1", "APOE_2", "ROH:APOE_1", "ROH:APOE_2"],
        {"ROH:APOE_genotype": ["ROH:APOE_1", "ROH:APOE_2"], "any_ROH": ["ROH", "ROH:APOE_1", "ROH:APOE_2"]},
        discrete_support=["APOE4"],
        description="Usually unsupported: very few ROH carriers are e4/e4")
    for value in (0, 1):
        add(f"A6_stratum_carrier{value}", f"ROH + covariates within e4 carrier={value}", "apoe", core,
            subset={"APOE_CARRIER": value})
    models.append(Model("A7_Fisher", "Fisher exact 2x2 (unadjusted)", "apoe", ["ROH"], {"ROH": ["ROH"]}, method="fisher"))

    # ---------------------------------------------------------------- global ancestry
    add("G0_Q_instead_of_PCs", "ROH + age + sex + APOE + global ancestry (no PCs)", "global_ancestry",
        no_pc + ["APOE4"] + q_nonref, {"ROH": ["ROH"], "global_ancestry": q_nonref},
        description=f"Global ancestry replaces PCs; reference component {reference}")
    q_grid = np.round(np.arange(0, 10.01, 0.25), 2)
    for a in ancestries:
        q = "Q10_" + a
        add(f"G1_{a}_adjust", f"ROH + covariates + APOE + global {a}", "global_ancestry", core + ["APOE4", q],
            {"ROH": ["ROH"], q: [q]})
        add(f"G2_{a}_APOExQ", f"APOE e4 x global {a}", "global_ancestry", core + ["APOE4", q, "APOE4:" + q],
            {"APOE4:" + q: ["APOE4:" + q]},
            contrasts=_slope(f"APOE4_by_Q_{a}", "APOE4", "APOE4:" + q, q, q_grid),
            description="Ancestry-specific APOE e4 effect (genome-wide ancestry as modifier)")
        add(f"G3_{a}_ROHxQ", f"ROH x global {a}", "global_ancestry", core + ["APOE4", q, "ROH:" + q],
            {"ROH:" + q: ["ROH:" + q], "any_ROH": ["ROH", "ROH:" + q]}, varying_support=[q],
            contrasts=_slope(f"ROH_by_Q_{a}", "ROH", "ROH:" + q, q, q_grid))

    # ---------------------------------------------------------------- local ancestry at the ROH locus
    la_grid = np.round(np.arange(0, 2.001, 0.05), 3)
    for a in ancestries:
        la, q = "LAD_" + a, "Q10_" + a
        add(f"L1_{a}_local", f"ROH + covariates + APOE + local {a} + global {a}", "local_ancestry",
            core + ["APOE4", q, la], {"ROH": ["ROH"], la: [la]},
            description="Primary local-ancestry model (admixture-mapping style: local conditional on global)")
        add(f"L1n_{a}_local_noglobal", f"as L1 without global {a}", "local_ancestry",
            core + ["APOE4", la], {"ROH": ["ROH"], la: [la]},
            description="Sensitivity: PCs only, shows how much of the local signal is genome-wide ancestry")
        add(f"L2_{a}_ROHxLA", f"ROH x local {a}", "local_ancestry", core + ["APOE4", q, la, "ROH:" + la],
            {"ROH:" + la: ["ROH:" + la], "any_ROH": ["ROH", "ROH:" + la]}, varying_support=[la],
            contrasts=_slope(f"ROH_by_LA_{a}", "ROH", "ROH:" + la, la, la_grid))
        add(f"L3_{a}_ROHxAPOE_LA", f"ROH x APOE adjusted for local+global {a}", "local_ancestry",
            core + ["APOE4", "ROH:APOE4", q, la], {"ROH:APOE4": ["ROH:APOE4"]}, varying_support=["APOE4"],
            contrasts=_slope(f"ROH_by_APOE4_LA_{a}", "ROH", "ROH:APOE4", "APOE4", [0, 1, 2]))
        o = "ROH_O_" + a
        add(f"L5_{a}_origin", f"ROH haplotype of {a} origin vs other origins", "local_ancestry",
            core + ["APOE4"] + q_nonref + [o], {o: [o], "any_ROH": ["ROH", o]}, carrier_split=[o],
            contrasts=[dict(group="ROH_by_origin", modifier=a, value=1.0, weights={"ROH": 1.0, o: 1.0}),
                       dict(group="ROH_by_origin", modifier="not_" + a, value=0.0, weights={"ROH": 1.0})])
    la_nonref = ["LAD_" + a for a in nonref]
    add("L4_joint_local", f"Joint local ancestry ({'+'.join(nonref)} vs {reference})", "local_ancestry",
        core + ["APOE4"] + q_nonref + la_nonref, {"ROH": ["ROH"], "local_ancestry_joint": la_nonref},
        description="Joint (k-1 df) test that local ancestry at the locus matters, given global ancestry")

    # ---------------------------------------------------------------- local ancestry at APOE (optional)
    if apoe_locus:
        for a in ancestries:
            laa, q = "LAA_" + a, "Q10_" + a
            add(f"E1_{a}_APOExLAapoe", f"APOE e4 x local {a} at APOE", "apoe_locus_ancestry",
                core + ["APOE4", q, laa, "APOE4:" + laa], {"APOE4:" + laa: ["APOE4:" + laa]},
                contrasts=_slope(f"APOE4_by_LAapoe_{a}", "APOE4", "APOE4:" + laa, laa, [0, 1, 2]))

    # ---------------------------------------------------------------- relatedness / kinship
    p1, p2 = core + ["APOE4"], core + ["APOE4", "ROH:APOE4"]
    p2_tests = {"ROH:APOE4": ["ROH:APOE4"]}
    add("K1_P1_GLMM", "Model 1, logistic mixed model with kinship (GRM) random effect", "relatedness", p1,
        method="glmm", description="PQL/REML GLMM (GMMAT-type); score test; requires --kinship")
    add("K2_P2_GLMM", "Model 2, logistic mixed model with kinship random effect", "relatedness", p2, p2_tests,
        method="glmm", varying_support=["APOE4"],
        contrasts=_slope("ROH_by_APOE4_GLMM", "ROH", "ROH:APOE4", "APOE4", [0, 1, 2]))
    add("K3_P1_robust", "Model 1, Firth with family-cluster-robust SE", "relatedness", p1, method="firth_robust")
    add("K4_P2_robust", "Model 2, Firth with family-cluster-robust SE", "relatedness", p2, p2_tests,
        method="firth_robust", varying_support=["APOE4"])
    add("K5_P1_unrelated", "Model 1, maximal unrelated subset", "relatedness", p1, sample="unrelated")
    add("K6_P2_unrelated", "Model 2, maximal unrelated subset", "relatedness", p2, p2_tests, sample="unrelated",
        varying_support=["APOE4"])
    add("K7_P1_FROH", "Model 1 + genome-wide F_ROH (inbreeding)", "relatedness", p1 + ["FROH_PCT"],
        {"ROH": ["ROH"], "FROH_PCT": ["FROH_PCT"]},
        description="Is the locus effect more than genome-wide autozygosity? F_ROH excludes the focal region")
    add("K8_P1_FROH_GLMM", "Model 1 + F_ROH, GLMM with kinship", "relatedness", p1 + ["FROH_PCT"], method="glmm")
    if primary == "glmm":
        swap = {"P1_ROH_cov_APOE": "relatedness", "P2_ROHxAPOE": "relatedness", "K1_P1_GLMM": "primary", "K2_P2_GLMM": "primary"}
        for m in models:
            if m.name in swap:
                for test in m.tests:
                    if m.test_family.get(test) != "apoe":
                        m.test_family[test] = swap[m.name]
    return models
