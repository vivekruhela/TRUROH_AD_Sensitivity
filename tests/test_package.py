from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from synthetic import make_run  # noqa: E402
from trurohassoc import apoe, catalog, common, firth, glmm  # noqa: E402
from trurohassoc import relatedness as relmod  # noqa: E402
from trurohassoc.data import load_run, region_frame  # noqa: E402
from trurohassoc.fit import build_design, fit_model  # noqa: E402

ENTRY = [sys.executable, str(ROOT / "roh_assoc.py")]
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")


def fit_args(**kw):
    base = dict(min_carriers=3, min_interaction_cell=5, max_fit_seconds=60, max_iterations=1000, tolerance=1e-5,
                prediction_draws=200)
    base.update(kw)
    return argparse.Namespace(**base)


class Statistics(unittest.TestCase):
    def test_separation_analytic_odds_ratio(self):
        X = np.column_stack([np.ones(20), np.r_[np.ones(10), np.zeros(10)]])
        full = firth.firth_optimize(X, X[:, 1])
        self.assertAlmostEqual(math.exp(full["beta"][1]), 441, places=3)
        lr, p, null = firth.likelihood_test(X, X[:, 1], full, [1])
        self.assertLess(p, .001)

    def test_joint_test_against_analytic_saturated_model(self):
        totals, events = np.array([20, 30, 25]), np.array([3, 18, 11])
        group = np.repeat(np.arange(3), totals)
        X = np.column_stack([np.ones(len(group)), group == 1, group == 2]).astype(float)
        y = np.concatenate([np.r_[np.ones(c), np.zeros(n - c)] for n, c in zip(totals, events)])
        lr, p, _ = firth.likelihood_test(X, y, firth.firth_optimize(X, y), [1, 2])
        probs, p0 = (events + .5) / (totals + 1), (events.sum() + 1.5) / (totals.sum() + 3)
        pen = lambda q: np.sum(events * np.log(q) + (totals - events) * np.log1p(-q)) + .5 * np.log(np.prod(totals * q * (1 - q)))  # noqa: E731
        self.assertAlmostEqual(lr, 2 * (pen(probs) - pen(p0)), places=6)
        self.assertAlmostEqual(p, chi2.sf(lr, 2), places=7)

    def test_score_matches_numerical_gradient(self):
        rng = np.random.default_rng(128)
        X = np.column_stack([np.ones(100), rng.normal(size=(100, 3))])
        y = rng.binomial(1, .4, 100)
        beta = rng.normal(0, .3, 4)
        _, score, _ = firth.firth_objective(beta, X, y)
        num = [(firth.firth_objective(beta + e, X, y)[0] - firth.firth_objective(beta - e, X, y)[0]) / 2e-6
               for e in np.eye(4) * 1e-6]
        np.testing.assert_allclose(score, num, atol=1e-6)

    def test_standardisation_back_transform_is_exact(self):
        rng = np.random.default_rng(5)
        n = 400
        d = pd.DataFrame({"ROH": (rng.random(n) < .15).astype(float), "APOE4": rng.choice([0, 1, 2], n, p=[.7, .25, .05]).astype(float),
                          "AGE_C": rng.normal(0, 1, n), "Q10_AFR": rng.uniform(0, 8, n)})
        d["AD"] = rng.binomial(1, expit(-.3 + .5 * d.APOE4 + .3 * d.ROH * d.APOE4 + .1 * d.Q10_AFR))
        terms = ["ROH", "APOE4", "AGE_C", "Q10_AFR", "ROH:APOE4"]
        X, kept, tf, _ = build_design(d, terms)
        full = firth.firth_optimize(X, d.AD.to_numpy(float))
        scale = np.array([1] + [tf[t][1] for t in kept[1:]])
        raw = np.column_stack([np.ones(n)] + [np.prod([d[c] for c in t.split(":")], axis=0) for t in terms])
        direct = firth.firth_optimize(raw, d.AD.to_numpy(float), max_iter=5000)
        np.testing.assert_allclose((full["beta"] / scale)[1:], direct["beta"][1:], atol=2e-4)

    def test_multiplicity_includes_failed_tests(self):
        out = firth.adjust_all_tests([.01, .04, np.nan])
        np.testing.assert_allclose(out["HOLM"], [.03, .08, np.nan], equal_nan=True)
        np.testing.assert_allclose(out["BH"], [.03, .06, np.nan], equal_nan=True)

    def test_no_opaque_apoe_inference(self):
        for value in ["0", "1", "2", "4", "C0", "-9"]:
            self.assertTrue(np.isnan(apoe.decode_genotype(value)))
        self.assertEqual(apoe.decode_genotype("e3/e4"), 1)
        self.assertEqual(apoe.decode_genotype("44"), 2)


class Kinship(unittest.TestCase):
    def test_glmm_without_relatedness_matches_logistic_ml(self):
        rng = np.random.default_rng(11)
        n = 500
        X = np.column_stack([np.ones(n), rng.normal(size=n), (rng.random(n) < .2).astype(float)])
        y = rng.binomial(1, expit(X @ [-.3, .5, .4])).astype(float)
        fit = glmm.fit(X, y, np.eye(n))
        b = np.zeros(3)
        for _ in range(50):                                   # plain ML logistic (IRLS)
            p = expit(X @ b)
            b = b + np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (y - p))
        self.assertLess(fit["tau"], 0.05)
        np.testing.assert_allclose(fit["beta"], b, atol=0.03)

    def test_glmm_recovers_family_variance_and_robust_singletons(self):
        rng = np.random.default_rng(3)
        fam = np.repeat(np.arange(300), 4)
        K = (fam[:, None] == fam[None, :]) * 0.5 + np.eye(len(fam)) * 0.5
        X = np.column_stack([np.ones(len(fam)), rng.normal(size=len(fam))])
        y = rng.binomial(1, expit(X @ [0, .5] + rng.normal(0, 1.5, 300)[fam])).astype(float)
        self.assertGreater(glmm.fit(X, y, K)["tau"], 0.5)
        full = firth.firth_optimize(X, y)
        rob, G = glmm.cluster_robust_cov(X, y, full["beta"], full["covariance"], np.arange(len(y)))
        u = X * (y - expit(X @ full["beta"]))[:, None]
        hc = full["covariance"] @ (u.T @ u) @ full["covariance"]
        self.assertEqual(G, len(y))
        np.testing.assert_allclose(rob, hc * G / (G - 1) * (len(y) - 1) / (len(y) - 2), rtol=1e-10)

    def test_unrelated_set_and_king_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.kin0"
            path.write_text("#FID1\tIID1\tFID2\tIID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
                            "f\tA\tf\tB\t1\t0\t0\t0.25\nf\tB\tf\tC\t1\t0\t0\t0.24\n"
                            "g\tD\tg\tE\t1\t0\t0\t0.01\n")
            pheno = pd.DataFrame({"FID": list("ffggh"), "IID": ["A", "B", "C", "D", "E"]})
            rel, summary = relmod.load_relatedness(path, pheno)
            self.assertEqual(summary["pairs_1st degree"], 2)
            self.assertAlmostEqual(rel.grm[0, 1], 0.5)
            self.assertEqual(rel.grm[3, 4], 0.0)              # below sparse threshold
            keep = rel.unrelated(["A", "B", "C", "D", "E"], {"A": 2, "C": 0})
            self.assertEqual(keep, {"A", "C", "D", "E"})       # B (two relatives) removed
            self.assertEqual(rel.cluster["A"], rel.cluster["C"])

    def test_gcta_grm_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "g"
            ids = ["A", "B", "C"]
            full = np.array([[1, .5, 0], [.5, 1, .02], [0, .02, 1]], dtype=np.float32)
            Path(str(prefix) + ".grm.id").write_text("".join(f"F{i}\t{i}\n" for i in ids))
            full[np.tril_indices(3)].tofile(str(prefix) + ".grm.bin")
            rel, summary = relmod.load_relatedness(Path(str(prefix) + ".grm.bin"), pd.DataFrame({"FID": ids, "IID": ids}))
            np.testing.assert_allclose(rel.grm, full, atol=1e-6)
            self.assertEqual(summary["pairs_1st degree"], 1)

    def test_froh_union_and_focal_exclusion(self):
        seg = pd.DataFrame({"IID": ["A", "A", "A", "B"], "CHR": [1, 1, 2, 6], "START": [0, 50, 0, 100],
                            "END": [100, 150, 10, 200]}).sort_values(["IID", "CHR", "START"])
        f = relmod.froh_excluding(seg, ["A", "B"])
        self.assertAlmostEqual(f["A"], 100 * 160 / relmod.AUTOSOME_BP)    # 0-150 union + 10
        f = relmod.froh_excluding(seg, ["A", "B"], 6, 150, 160)
        self.assertEqual(f["B"], 0.0)

    def test_primary_switch(self):
        models = {m.name: m for m in catalog.model_catalog(["AGE_C", "SEX"], ["EUR", "AFR"], "EUR", primary="glmm")}
        self.assertEqual(models["K1_P1_GLMM"].test_family, {"ROH": "primary"})
        self.assertEqual(models["P1_ROH_cov_APOE"].test_family["ROH"], "relatedness")
        self.assertEqual(models["P1_ROH_cov_APOE"].test_family["APOE4"], "apoe")


class Catalog(unittest.TestCase):
    def setUp(self):
        self.core = ["AGE_C", "SEX", "PC1", "PC2", "PC3", "PC4", "PC5"]
        self.models = catalog.model_catalog(self.core, ["NAM", "EUR", "AFR"], "EUR", apoe_locus=True)

    def test_age_sex_pcs_never_interact(self):
        for m in self.models:
            for term in m.terms:
                if ":" in term:
                    self.assertFalse(set(term.split(":")) & set(self.core), (m.name, term))

    def test_interactions_are_hierarchical(self):
        for m in self.models:
            for term in m.terms:
                if ":" in term:
                    self.assertTrue(set(term.split(":")).issubset(m.terms), (m.name, term))

    def test_requested_models_present(self):
        names = {m.name: m for m in self.models}
        self.assertEqual(names["P1_ROH_cov_APOE"].terms, ["ROH"] + self.core + ["APOE4"])
        self.assertIn("ROH:APOE4", names["P2_ROHxAPOE"].terms)
        self.assertIn("Q10_NAM", names["L1_NAM_local"].terms)          # local with global
        self.assertNotIn("Q10_NAM", names["L1n_NAM_local_noglobal"].terms)
        self.assertEqual({m.family for m in self.models}, set(catalog.FAMILIES))
        self.assertEqual(len(self.models), 49)
        self.assertEqual(sum(len(m.tests) for m in self.models), 76)
        self.assertEqual(names["P1_ROH_cov_APOE"].test_family, {"APOE4": "apoe"})


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="truroh v2 ")
        cls.base = Path(cls.tmp.name)
        cls.rundir, cls.cfg, cls.pheno, cls.region = make_run(cls.base, n=900, n_carriers=60, seed=7)
        cls.before = {str(p.relative_to(cls.rundir)): common.sha256(p) for p in cls.rundir.rglob("*") if p.is_file()}
        cls.full = cls.base / "full analysis"
        cls.result = subprocess.run(ENTRY + ["--run-dir", str(cls.rundir), "--apoe-map", str(cls.base / "map.json"),
                                             "--apoe-locus-la", str(cls.base / "apoe_locus_la.tsv"),
                                             "--kinship", str(cls.base / "cohort.kin0"),
                                             "--prediction-draws", "200", "--out-dir", str(cls.full)],
                                    capture_output=True, text=True, env=ENV)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_end_to_end_outputs(self):
        self.assertEqual(self.result.returncode, 0, self.result.stdout[-2000:] + self.result.stderr)
        tests = pd.read_csv(self.full / "tables/all_association_tests.tsv", sep="\t")
        self.assertEqual(len(tests), 76 + 3)
        k1 = tests[(tests.MODEL == "K1_P1_GLMM")].iloc[0]
        self.assertEqual(k1.STATUS, "converged")
        self.assertGreaterEqual(k1.GLMM_TAU, 0)
        self.assertTrue(tests[tests.MODEL.str.startswith("K")].STATUS.eq("converged").all())
        self.assertTrue((self.full / "tables/relatedness_diagnostics.tsv").is_file())
        p1 = tests[(tests.MODEL == "P1_ROH_cov_APOE") & (tests.TEST == "ROH")].iloc[0]
        self.assertEqual(p1.STATUS, "converged")
        self.assertEqual(p1.FAMILY, "primary")
        self.assertEqual(tests[(tests.MODEL == "P1_ROH_cov_APOE") & (tests.TEST == "APOE4")].FAMILY.iloc[0], "apoe")
        self.assertEqual(int((tests.FAMILY == "primary").sum()), 3)
        self.assertTrue(tests.HOLM_P_ALL.dropna().ge(tests.P_VALUE.dropna().loc[tests.HOLM_P_ALL.dropna().index]).all())
        for stem in ["Fig1_ROH_main_effect", "Fig2_APOE_and_ROHxAPOE", "Fig3_ancestry_effects", "FigS1_data_overview"]:
            for ext in ["pdf", "svg", "png", "tiff"]:
                self.assertTrue(list((self.full / "figures").glob(f"{stem}_*.{ext}")), (stem, ext))
        self.assertTrue((self.full / "figures/FigS2_all_tests_overview.pdf").is_file())
        page = (self.full / "report.html").read_text()
        self.assertIn("Should global ancestry be included", page)
        self.assertIn("data:image/png;base64", page)
        self.assertEqual(common.read_json(self.full / "analysis_status.json")["state"], "completed")

    def test_contrast_at_zero_equals_main_coefficient(self):
        k = pd.read_csv(self.full / "tables/effect_contrasts.tsv", sep="\t")
        c = pd.read_csv(self.full / "tables/all_coefficients.tsv", sep="\t")
        at0 = k[(k.GROUP == "ROH_by_APOE4") & (k.VALUE == 0)].BETA.iloc[0]
        main = c[(c.MODEL == "P2_ROHxAPOE") & (c.TERM == "ROH")].BETA.iloc[0]
        self.assertAlmostEqual(at0, main, places=10)
        preds = pd.read_csv(self.full / "tables/predicted_risk.tsv", sep="\t")
        self.assertEqual(len(preds), 6)
        self.assertTrue(((preds.RISK_LOW <= preds.RISK) & (preds.RISK <= preds.RISK_HIGH)).all())

    def test_source_run_unchanged(self):
        after = {str(p.relative_to(self.rundir)): common.sha256(p) for p in self.rundir.rglob("*") if p.is_file()}
        self.assertEqual(self.before, after)

    def test_apoe_decoded_and_audited(self):
        audit = common.read_json(self.full / "apoe_audit.json")
        self.assertEqual(audit["status"], "available")
        self.assertGreater(audit["dosage_available_N"], 800)
        self.assertEqual(audit["warnings"], [])
        counts = pd.read_csv(self.full / "apoe_source_value_counts.tsv", sep="\t", dtype=str, keep_default_na=False)
        self.assertEqual(set(counts.DECODED_AS), {"0", "1", "2", "missing"})

    def test_without_map_apoe_models_reported_not_dropped(self):
        out = self.base / "no map"
        r = subprocess.run(ENTRY + ["--run-dir", str(self.rundir), "--out-dir", str(out), "--prediction-draws", "50"],
                           capture_output=True, text=True, env=ENV)
        self.assertEqual(r.returncode, 0, r.stderr)
        tests = pd.read_csv(out / "tables/all_association_tests.tsv", sep="\t")
        self.assertTrue(tests.loc[tests.MODEL == "P1_ROH_cov_APOE", "STATUS"].eq("no_complete_samples").all())
        self.assertTrue(tests.loc[tests.MODEL == "K1_P1_GLMM", "STATUS"].eq("unavailable_kinship").all())
        self.assertIn("APOE ε4 dosage is unavailable", (out / "report.html").read_text())

    def test_unmapped_value_and_reversed_map(self):
        out = self.base / "audits"
        out.mkdir(exist_ok=True)
        _, pheno, *_ = load_run(self.rundir, out)
        bad = self.base / "bad.json"
        common.write_json(bad, {"encoding": "epsilon4_dosage", "mapping": {"0": 0, "1": 1}})
        with self.assertRaisesRegex(ValueError, "Unmapped APOE"):
            apoe.prepare_apoe(pheno, self.cfg, out, bad)
        common.write_json(bad, {"encoding": "epsilon4_dosage", "mapping": {"0": 2, "1": 1, "2": 0}})
        _, audit = apoe.prepare_apoe(pheno, self.cfg, out, bad)
        self.assertTrue(any("Implausible" in w or "reversed" in w for w in audit["warnings"]))

    def test_origin_and_local_ancestry_frame(self):
        out = self.base / "frame"
        out.mkdir(exist_ok=True)
        cfg, pheno, core, hc, members, local, *_ = load_run(self.rundir, out, self.base / "map.json")
        d = region_frame(pheno, next(hc.itertuples(index=False)), members, local, ["NAM", "EUR", "AFR"])
        carriers = d[d.ROH.eq(1)]
        self.assertTrue(np.isin(carriers.LAD_NAM, [0, 2]).all())   # homozygous ancestry
        self.assertEqual(int(carriers[["ROH_O_NAM", "ROH_O_EUR", "ROH_O_AFR"]].sum(axis=1).sum()), len(carriers))
        self.assertTrue(d.loc[d.ROH.eq(0), "ROH_O_NAM"].eq(0).all())

    def test_audit_only_and_profile(self):
        prof = self.base / "p.json"
        common.write_json(prof, {"run_dir": str(self.rundir), "apoe_map": "@pkg/apoe_dosage_map.json",
                                 "out_parent": str(self.base / "parent"), "out_name": "X"})
        r = subprocess.run(ENTRY + ["--profile", str(prof), "--audit-only"], capture_output=True, text=True, env=ENV)
        self.assertEqual(r.returncode, 0, r.stderr)
        made = list((self.base / "parent").glob("X_*"))
        self.assertEqual(len(made), 1)
        self.assertEqual(common.read_json(made[0] / "analysis_status.json")["state"], "audit_completed")
        self.assertEqual(len(pd.read_csv(made[0] / "tables/model_plan.tsv", sep="\t")), 73)  # no APOE-locus file
        self.assertTrue(common.read_json(made[0] / "apoe_audit.json")["mapping_file"].endswith("apoe_dosage_map.json"))

    def test_profile_column_names_stay_strings(self):
        # regression: GAPP profile with apoe_column/apoe_fid_column/apoe_iid_column (v2.0.0 turned them into paths)
        prof = self.base / "gapp_like.json"
        common.write_json(prof, {"run_dir": str(self.rundir), "apoe_column": "APOE_reported", "apoe_fid_column": "FID",
                                 "apoe_iid_column": "IID", "apoe_map": "@pkg/apoe_dosage_map.json",
                                 "out_parent": str(self.base / "gapp_parent"), "out_name": "G"})
        r = subprocess.run(ENTRY + ["--profile", str(prof), "--audit-only"], capture_output=True, text=True, env=ENV)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("FID='FID'; IID='IID'; APOE='APOE_reported'", r.stdout)
        made = list((self.base / "gapp_parent").glob("G_*"))[0]
        self.assertEqual(common.read_json(made / "apoe_audit.json")["dosage_available_N"], 
                         int(pd.read_csv(self.base / "source phenotype.tsv", sep="\t", dtype=str).APOE_reported.isin(["0", "1", "2"]).sum()))

    def test_wrong_apoe_locus_file_is_explained(self):
        bad = self.base / "rfmix_sample_map.tsv"
        bad.write_text("HG00096\tEUR\nHG00097\tEUR\n")
        r = subprocess.run(ENTRY + ["--run-dir", str(self.rundir), "--apoe-locus-la", str(bad), "--audit-only"],
                           capture_output=True, text=True, env=ENV)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("make_apoe_locus_la.py", r.stderr)

    def test_bundled_profiles_parse(self):
        for name in ["ch", "gapp"]:
            prof = common.read_json(ROOT / "profiles" / f"{name}.json")
            self.assertTrue(set(prof) <= {"config", "run_dir", "out_dir", "out_parent", "out_name", "apoe_map", "apoe_column",
                                          "apoe_source", "apoe_fid_column", "apoe_iid_column", "apoe_locus_la",
                                          "min_carriers", "min_interaction_cell", "description"})

    def test_rejects_output_inside_run(self):
        r = subprocess.run(ENTRY + ["--run-dir", str(self.rundir), "--out-dir", str(self.rundir / "x"), "--audit-only"],
                           capture_output=True, text=True, env=ENV)
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.rundir / "x").exists())

    def test_override_requires_map(self):
        r = subprocess.run(ENTRY + ["--run-dir", str(self.rundir), "--apoe-column", "APOE_reported", "--audit-only"],
                           capture_output=True, text=True, env=ENV)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("require --apoe-map", r.stderr)

    def test_submit_and_spooled_worker(self):
        mock = self.base / "bin"
        mock.mkdir(exist_ok=True)
        cap = self.base / "sbatch.json"
        (mock / "sbatch").write_text("#!" + sys.executable + "\nimport json,os,sys\nopen(os.environ['CAPTURE'],'w').write(json.dumps(sys.argv[1:]))\n"
                                     "print('sbatch: error: fixture rejection', file=sys.stderr)\nsys.exit(1)\n")
        (mock / "sbatch").chmod(0o755)
        env = dict(ENV, PATH=str(mock) + os.pathsep + ENV["PATH"], CAPTURE=str(cap))
        for k in ("TRUROH_AD_CONSTRAINT", "TRUROH_CONSTRAINT", "TRUROH_AD_PARTITION", "TRUROH_AD_ACCOUNT"):
            env.pop(k, None)
        job = ["--run-dir", str(self.rundir), "--apoe-map", str(self.base / "map.json"), "--out-dir", str(self.base / "queued")]
        # default: no partition/account/constraint options at all
        r = subprocess.run(["bash", str(ROOT / "submit.sh")] + job, cwd=self.base, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("fixture rejection", r.stderr)
        args = json.loads(cap.read_text())
        self.assertFalse(any(a.startswith(("--constraint", "--partition", "--account")) for a in args), args)
        self.assertEqual(args[-(len(job) + 2):], [str(ROOT / "slurm/job.sbatch"), str(ROOT)] + job)
        # configured site options are passed through
        env2 = dict(env, TRUROH_AD_CONSTRAINT="cpu8mem64a", TRUROH_AD_PARTITION="short", TRUROH_AD_MEMORY="8G")
        subprocess.run(["bash", str(ROOT / "submit.sh")] + job, cwd=self.base, env=env2, capture_output=True, text=True)
        args = json.loads(cap.read_text())
        for opt in ("--constraint=cpu8mem64a", "--partition=short", "--mem=8G"):
            self.assertIn(opt, args)
        # dry run prints the command and never calls sbatch
        cap.unlink()
        r = subprocess.run(["bash", str(ROOT / "submit.sh"), "--dry-run"] + job, cwd=self.base, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("sbatch", r.stdout)
        self.assertFalse(cap.exists())
        # pre-submission check rejects a missing run before queueing
        r = subprocess.run(["bash", str(ROOT / "submit.sh"), "--run-dir", str(self.base / "nope"), "--audit-only"],
                           cwd=self.base, env=env, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("completed TRU-ROH run not found", r.stderr)
        self.assertFalse(cap.exists())
        spool = self.base / "spool"
        spool.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "slurm/job.sbatch", spool / "slurm_script")
        out = self.base / "spooled"
        r = subprocess.run(["bash", str(spool / "slurm_script"), str(ROOT), "--run-dir", str(self.rundir), "--audit-only",
                            "--out-dir", str(out)], cwd=spool, capture_output=True, text=True, env=ENV)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
