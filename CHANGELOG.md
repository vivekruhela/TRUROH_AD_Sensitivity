# Changes

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

