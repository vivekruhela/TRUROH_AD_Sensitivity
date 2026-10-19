# Changes

## 2.1.0 — 2026-10-19 UTC

Primary ancestry `auto`, and removal of all multiple-testing correction.

Primary ancestry:

- New `--primary-ancestry` (profile key `primary_ancestry`, wizard question): `auto` selects the ancestry with
  the highest mean global ancestry over the prepared cohort (outcome-free; top-two margin < 0.05 flagged as a
  close call); an explicit label is kept as the pre-specified hypothesis and noted when it is not the majority.
  Default: the source run's `association.primary_ancestry`, else `auto`. Without global ancestry, `auto` uses
  the run's `results/primary_ancestry_selection.json` (TRUROH_Ancestry >= 2.2.0), else the first ancestry.
- The primary ancestry is listed first (catalog, tables, figures); the report shows it as a KPI and in key
  results (L1 local-ancestry and L5 haplotype-origin estimates). Recorded in `input_audit.json` and
  `summary.json`. The set of models and tests is unchanged.
- Tests: auto / explicit / alias / invalid selection; legacy correction columns not imported (30 tests).

Multiple testing:

Removed all multiple-testing correction. P values are nominal (unadjusted); no multiple-testing correction is
applied because the number of individuals and ROH regions gives no power after correction. Interpret associations
as exploratory and look for replication across cohorts. Raw P values, ORs, CIs, statuses, the model catalog and
all fits are unchanged.

- `firth.adjust_all_tests` (Holm / Bonferroni / Benjamini–Hochberg) deleted.
- Columns `HOLM_P_ALL`, `BONFERRONI_P_ALL`, `BH_Q_ALL` and `HOLM_P_FAMILY` are no longer written to
  `tables/all_association_tests.tsv` or shown in the report tables; `NOMINAL_P_LT_0_05` and `N_TESTS_ALL` remain.
- `summary.json`: `primary_holm_tests` and `catalog_holm_tests` removed; `primary_nominal_tests` added.
- Console summary now ends "Completed: N tests; M converged; K nominal p<0.05 (J of P primary)."
- `FigS2_all_tests_overview`: Bonferroni reference line removed; only the nominal P = 0.05 line is drawn.
- Report key results, captions, methods text, limitations and `interpretation.md`, the catalog family
  description, README and the synthetic example report now describe nominal P values.
- `Q_VALUE`, `BONFERRONI_P` and `FDR_ASSOCIATED` imported from older TRUROH `results/associations.tsv` are dropped.
- Tests: the Holm/BH unit test was removed; the end-to-end test now checks that no correction columns or
  summary keys are written and that `NOMINAL_P_LT_0_05` and the nominal counts match `P_VALUE`.

## 2.0.0 public packaging — 2026-10-09

No change to the analysis code (`trurohassoc/`, whose SHA-256 values are recorded in each `analysis_manifest.json`).

- `configure.sh` wizard (`tools/configure.py`): inspects the completed TRU-ROH run and the APOE column, writes a profile and `config/site.env`.
- `submit.sh`: partition/account/constraint are optional and omitted when empty (previously a site-specific constraint was always passed); new `--dry-run`, `TRUROH_AD_*` settings, `TRUROH_AD_DEPENDENCY`, and a pre-submission check (`tools/check_args.py`).
- The SLURM job sources `site.env` (modules/conda).
- Site-specific output paths removed from the bundled CH/GAPP profiles; an internal CH review note was removed.
- End-to-end test with an `sbatch` emulator (`tests/e2e`); `environment.yml`, `.gitignore`, `CITATION.cff`, CI workflow; README rewritten.

## 2.0.0 — 2026-10-08

Changes relative to v1.0.x:

| Problem | v2.0.0 |
|---|---|
| `APOE_reported` holds bare 0/1/2 codes, which are (correctly) never guessed, so APOE decoded N = 0 and every APOE model was skipped | Cohort profiles supply `apoe_dosage_map.json` (explicit `epsilon4_dosage` encoding) so the run decodes APOE automatically. The audit lists every source value and what it became, the ε4 allele frequency, and warns if the frequency is implausible or cases carry fewer ε4 alleles than controls (reversed map). |
| Two-step manual workflow (audit, edit map, rerun with several flags) | `--profile ch|gapp` runs everything; CLI flags still override. |
| Age and sex were used as interaction terms; many models answered questions not asked | Age, sex, PCs are adjustment covariates only. Effect modifiers are APOE and ancestry. |
| Only ROH coefficients exported; nuisance terms on a standardized scale | Every coefficient back-transformed exactly to natural units (per ε4 allele, per 10 % global ancestry, per haplotype of local ancestry, per decade of age). |
| No APOE or ancestry figures | Four main/supplementary figures + all-tests overview, Nature Methods conventions. |
| Local ancestry only as a ROH modifier at LA = 0/1 | Local ancestry as covariate, as modifier, and as **ancestral origin of the ROH haplotype**; global-ancestry decision quantified. |

