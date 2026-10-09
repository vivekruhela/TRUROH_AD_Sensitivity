# TRU-ROH AD sensitivity — ROH × APOE × ancestry association models

Region-level Alzheimer's disease (AD) association analysis of the **high-confidence ROH regions** found by
[TRUROH_Ancestry](https://github.com/vivekruhela/TRUROH_Ancestry) (locked GRID_ID 1433). One command:

1. decodes APOE (with an audit of every source value);
2. fits a fixed, pre-registered catalog of models — ROH main effects, ROH × APOE, global and local
   ancestry, haplotype ancestral origin, and kinship-aware GLMM / cluster-robust / unrelated-subset
   versions;
3. writes tables, publication figures (PDF/SVG/600-dpi PNG/TIFF) and a self-contained HTML report.

The completed TRU-ROH run is read-only: nothing in it is changed and no ROH stage is rerun. The
simulation framework that fixed the ROH thresholds is
[TRUROH_Simulations](https://github.com/vivekruhela/TRUROH_Simulations).

> **Reporting in the paper.** Cohort sizes give little power after multiple-testing correction, so the
> manuscript reports **nominal** associations (P < 0.05) and looks for models that replicate across cohorts
> (Caribbean Hispanic and Peruvian). The package still writes Holm/Bonferroni/BH-adjusted values for
> every test (`tables/all_association_tests.tsv`) so that readers can apply any correction.

## Requirements

* A **completed** TRUROH_Ancestry run (`final_status.json` state `completed`)
* Python ≥ 3.10 with numpy, pandas, scipy, matplotlib (`environment.yml`)
* SLURM for cluster submission (optional — `run.sh` runs directly on any node)
* Optional: a KING kinship table or GCTA GRM, and RFMix chr19 output for APOE-locus local ancestry

```bash
git clone https://github.com/vivekruhela/TRUROH_AD_Sensitivity.git
conda env create -f TRUROH_AD_Sensitivity/environment.yml && conda activate truroh-ad
```

## Quick start

```bash
cd TRUROH_AD_Sensitivity

# 1. configure (interactive): writes profiles/<name>.json and config/site.env
bash configure.sh

# 2. audit APOE decoding only (seconds) — check apoe_source_value_counts.tsv / apoe_audit.json
bash run.sh --profile my_cohort --audit-only

# 3. submit one SLURM job (or run.sh --profile my_cohort to run in the foreground)
bash submit.sh --dry-run --profile my_cohort
bash submit.sh --profile my_cohort
```

`configure.sh` asks for:

| Section | Questions |
|---|---|
| TRU-ROH run | the TRUROH_Ancestry cohort JSON or its output directory (state and number of high-confidence regions are shown) |
| APOE | source table (default: the phenotype file used by TRU-ROH); column; code map. Observed values are listed; bare `0/1/2` codes default to the bundled ε4-dosage map, genotype strings (`e3/e4`, `34`) need no map |
| Relatedness / optional | kinship file, primary family (`firth` or `glmm`, fixed before seeing results), own inbreeding estimate, APOE-locus local ancestry, minimum carriers |
| Output | parent directory and folder name (a UTC time stamp is appended; nothing is overwritten) |
| SLURM / software | partition, account, constraint (blank = omitted), CPUs, memory, time, modules, conda environment, Python |

Every answer is checked against the files. The profile is then validated with the same argument parser
and input checks the job uses (`tools/check_args.py`). `submit.sh` runs that same check before
queueing, so a wrong path fails immediately rather than in the queue. Non-interactive set-up:
`bash configure.sh --answers answers.json` (keys are shown in brackets in the prompts; see
`tests/e2e/run_e2e.sh`).

Without the wizard, everything can be given on the command line (CLI flags override a profile):

```bash
bash run.sh --config /path/TRUROH_Ancestry/config/my_cohort.json \
            --apoe-map apoe_dosage_map.json --kinship /path/cohort.kin0 --out-parent /path/results
```

### Scheduler settings

`submit.sh` reads `config/site.env` (or the file named by `TRUROH_AD_SITE_ENV`) and the environment:

| Variable | Default |
|---|---|
| `TRUROH_AD_PARTITION`, `TRUROH_AD_ACCOUNT`, `TRUROH_AD_CONSTRAINT` | empty → option omitted |
| `TRUROH_AD_CPUS`, `TRUROH_AD_MEMORY`, `TRUROH_AD_TIME` | 4, 16G, 1-00:00:00 |
| `TRUROH_AD_LOG_DIR` | `<package>/logs` |
| `TRUROH_AD_DEPENDENCY` | e.g. `afterok:<TRU-ROH finalizer job id>` to queue before the ROH run has finished |
| `PYTHON_BIN` | `python3` |

The job sources the same `site.env`, so modules or a conda environment named in the wizard are active on
the compute node. A full run takes about 1–2 minutes per region (GLMM fits dominate) for ~2,300 people.

### Paper cohorts

`profiles/ch.json` and `profiles/gapp.json` are the profiles used for the Caribbean Hispanic and Peruvian
(GAPP/PeADI) analyses. Their `config` paths are relative to a TRUROH_Ancestry checkout, so run them from
that directory:

```bash
cd TRUROH_Ancestry
bash ../TRUROH_AD_Sensitivity/submit.sh --profile gapp --kinship /path/gapp_king.kin0
```

## Model catalog (fixed before fitting; 49 models / 76 tests per region with 3 ancestries and APOE-locus LA)

Covariates `COV` = age (centred, per 10 y) + sex + PC1–5 (from the saved run configuration).

| Family | Model | Formula (AD ~) | Tested |
|---|---|---|---|
| **primary** | P1 (your model 1) | ROH + COV + APOE4 | ROH |
| **primary** | P2 (your model 2) | ROH + COV + APOE4 + ROH×APOE4 | ROH×APOE4; ROH+ROH×APOE4 (2 df) |
| apoe | A0, A1, A1m | ROH; ROH + COV (original M1); ROH + COV on APOE-typed samples | ROH |
| apoe | A2, A3 | ε4 carrier; ε4 genotype (1 / 2 copies) instead of dosage | ROH, APOE |
| apoe | A4, A5 | ROH × carrier; ROH × genotype | interaction |
| apoe | A6 | ROH + COV within ε4 non-carriers / carriers | ROH |
| apoe | A7 | Fisher exact 2×2 | ROH |
| global_ancestry | G0 | ROH + age + sex + APOE4 + global ancestry (K−1 components) **instead of PCs** | ROH, global block |
| global_ancestry | G1_K | P1 + Q_K | ROH, Q_K |
| global_ancestry | G2_K | P1 + Q_K + **APOE4×Q_K** (ancestry-specific APOE effect) | APOE4×Q_K |
| global_ancestry | G3_K | P1 + Q_K + ROH×Q_K | ROH×Q_K |
| local_ancestry | L1_K | P1 + Q_K + LA_K (local given global; primary LA model) | ROH, LA_K |
| local_ancestry | L1n_K | P1 + LA_K (no global; sensitivity) | ROH, LA_K |
| local_ancestry | L2_K | L1_K + ROH×LA_K | ROH×LA_K |
| local_ancestry | L3_K | P2 + Q_K + LA_K (ROH×APOE robust to ancestry) | ROH×APOE4 |
| local_ancestry | L4 | P1 + Q (K−1) + LA (K−1), joint | joint LA block |
| local_ancestry | L5_K | P1 + Q (K−1) + ROH_origin=K (carriers whose ROH haplotype is K-ancestry vs other origins) | origin heterogeneity |
| apoe_locus_ancestry | E1_K (only with `--apoe-locus-la`) | P1 + Q_K + LAapoe_K + APOE4×LAapoe_K | APOE4×LAapoe_K |
| relatedness | K1–K8 | Models 1–2 as GLMM, cluster-robust, unrelated subset; Model 1 + F_ROH | ROH, ROH×APOE4, F_ROH |

K = each configured ancestry (NAM, EUR, AFR). Q_K is global ancestry per 10 percentage points; LA_K is the
number of K-ancestry haplotypes (0–2) at the ROH region (2 × the local-ancestry proportion in
`local_ancestry_for_association.tsv.gz`). Every interaction keeps all its lower-order terms.

Three-way ROH × APOE × ancestry models are deliberately not fitted: with ~56 carriers and few ε4 carriers
among them, the cells are empty or near-empty, and such a model would be noise.

## Kinship / relatedness

**Should kinship be in the ROH–AD models? Yes, whenever the cohort contains relatives, and the run tells you how many it
does.** But kinship is a property of *pairs*, so it cannot be entered as an ordinary covariate.
The package handles it in the three standard ways plus one covariate:

| Model | What it does |
|---|---|
| K1 / K2 — GLMM | Models 1 / 2 as logistic mixed models with a GRM random effect (PQL + REML, GMMAT-type score test) |
| K3 / K4 — cluster-robust | Firth estimates with CR1 sandwich SEs, relatives (≥ 2nd degree, connected components) as clusters |
| K5 / K6 — unrelated subset | Models 1 / 2 in a maximal set with no pair closer than 2nd degree (greedy; keeps ROH carriers when ties; never uses AD) |
| K7 / K8 — F_ROH | Model 1 + genome-wide autozygosity (F_ROH), Firth and GLMM. F_ROH is the kinship between a person's own parents, the individual-level relatedness quantity that *is* a covariate |

Why it matters for ROH in particular: relatives share the same IBD segment, so ROH carriers cluster in
families, and families also share AD risk. Treating relatives as independent then makes P values too small.
In a null simulation (`tests/validate_kinship.py`, 400 replicates, sibships with shared ROH and familial
risk, no true ROH effect) the type I error at α = 0.05 was **10.3 %** for Firth ignoring kinship, versus
4.8 % (cluster-robust), 6.0 % (GLMM) and 4.8 % (unrelated subset); see `KINSHIP_VALIDATION.json`.

F_ROH is computed automatically as the share of the autosomes covered by the union of all TRUROH candidate
ROH regions each person carries, **excluding regions overlapping the tested region** (otherwise it would
partly be the exposure itself). Supply `--inbreeding file.tsv` (IID, F as a 0–1 proportion, e.g. PLINK
`--homozyg` KB / 2,875,001 or `--het` F) to use your own estimate instead.

Kinship input (`--kinship`), either:

* a KING table — plink2 `--make-king-table` output (`#FID1 IID1 FID2 IID2 … KINSHIP`) or KING `.kin0`
  (`FID1 ID1 FID2 ID2 … Kinship`); pairs with kinship ≥ 0.0221 build a sparse GRM (negatives set to 0);
* a GCTA GRM (`prefix.grm.bin` + `prefix.grm.id`, e.g. plink2 `--make-grm-bin`).

```bash
# LD-pruned common autosomal SNPs from the same genotypes TRUROH used
plink2 --bfile ch_genotypes --autosome --maf 0.05 --indep-pairwise 200 50 0.2 --out ch_prune
plink2 --bfile ch_genotypes --extract ch_prune.prune.in --make-king-table \
       --king-table-filter 0.0221 --out ch_king          # -> ch_king.kin0
```

IIDs must match the phenotype IIDs. Without `--kinship`, GLMM models are reported as unavailable and the
robust / unrelated analyses fall back to FID families. The report has a relatedness section (pairs by degree,
how many ROH carriers have a carrier relative, independent carrier clusters, the unrelated-subset size) and
Fig. 1 shows all relatedness-aware estimates beside Model 1.

Primary family: by default the primary hypotheses are the Firth Models 1–2 (comparable to the original
run) and K1–K8 are sensitivity analyses. If the relatedness diagnostics show many related carriers, decide
**before looking at results** to make the GLMM versions primary with `--primary glmm` (requires `--kinship`);
the choice is recorded in `analysis_manifest.json`. GLMM fits add about 1–2 minutes per region for ~2,300 people.

## Should global ancestry be included with local ancestry?

**Yes, for the primary local-ancestry models; the PC-only version is kept as a sensitivity analysis.**
Local ancestry at a locus is a draw around each person's genome-wide ancestry. Without global ancestry,
the local-ancestry coefficient mixes the locus-specific signal with everything genome-wide ancestry
correlates with (polygenic background, environment, recruitment). Conditioning on global ancestry is the
standard admixture-mapping contrast. The price is collinearity, so the report quantifies it per region and
ancestry (`tables/ancestry_collinearity.tsv`: local–global correlation, R² of local on global + PCs, VIF)
and shows the local-ancestry estimate with and without global ancestry side by side (Fig. 3a). Global
components sum to one, so they are entered one at a time or as K−1 components with the configured
reference (EUR).

Because an ROH is homozygous by descent, a carrier's two haplotypes at the locus nearly always share one
ancestry (Fig. S1a). So "which ancestry plays a role" for the ROH is answered most directly by **L5**: the
AD odds ratio for carriers whose homozygous haplotype is Native American vs European vs African origin
(Fig. 3c). For APOE, the analogous question is answered by **G2** (APOE × global ancestry) and, if you
supply local ancestry at APOE, **E1** (Fig. 3d–e). This is association with AD status in a case-control
sample; it cannot address progression.

## Figures (`figures/`, each as PDF, SVG, 600-dpi PNG and TIFF)

| File | Content |
|---|---|
| `Fig1_ROH_main_effect_<region>` | ROH OR across all main-effect specifications, original estimates included; Model 1 highlighted |
| `Fig2_APOE_and_ROHxAPOE_<region>` | a APOE ε4 ORs; b ROH OR at 0/1/2 ε4 alleles + ε4 strata; c g-computed P(AD) by ROH × ε4 |
| `Fig3_ancestry_effects_<region>` | a global/local ancestry ORs (with/without global); b ROH OR vs local ancestry; c ROH OR by haplotype origin; d APOE OR vs global ancestry; e APOE OR by local ancestry at APOE; f ROH OR vs global ancestry |
| `FigS1_data_overview_<region>` | local-ancestry dosage by carrier status, local vs global ancestry, AD fraction by ε4 × ROH, carriers by origin |
| `FigS2_all_tests_overview` | −log10 P of every test by family with nominal and Bonferroni lines |

Style: 183 mm width, Arial/Helvetica (Liberation Sans fallback) 5–7 pt, embedded TrueType fonts (editable in
Illustrator), Okabe-Ito colours with every colour also labelled, OR on log axes, extrapolated estimates
shown open/dashed rather than hidden. Report captions are draft legends.

## Inference

* Firth penalized logistic regression; P from penalized likelihood-ratio tests that keep the full-design
  penalty; Wald 95 % CIs; contrasts (e.g. ROH OR at ε4 = 2) by the delta method; predicted risks by
  g-computation with 2,000 coefficient draws.
* Support rules fixed in advance: ≥3 carriers and non-carriers; ≥5 carriers per modifier level for
  interactions and origin models; otherwise the model is reported as not estimable (never dropped) and
  counts as P = 1 in multiplicity correction.
* Multiplicity: the three primary tests (P1 ROH, P2 interaction, P2 2-df) are Holm-corrected within the
  primary family. All other tests are exploratory and carry raw P, within-family Holm P and catalog-wide
  Holm/Bonferroni/BH values (`tables/all_association_tests.tsv`). The APOE main effect is a positive control,
  so it sits in the `apoe` family, not the primary family.
* Relatedness: see "Kinship / relatedness". The GLMM uses PQL, which can underestimate the kinship variance for
  binary traits; the score tests are the GMMAT/SAIGE standard but asymptotic.

## Inputs read from the completed run

`run_manifest.json`, `final_status.json`, `config.resolved.json`, `inputs/phenotypes_prepared.tsv`,
`results/regions_scored.tsv`, `results/members.tsv.gz`, and if present `results/global_ancestry.tsv`,
`results/local_ancestry_for_association.tsv.gz`, `results/associations.tsv` (original tests, imported).
APOE is read from the saved `phenotype_file` / `columns.apoe_genotype` unless the profile or CLI overrides
the source, column or ID headers.

### Optional: local ancestry at APOE (`--apoe-locus-la`)

Only the E1 models (APOE ε4 × local ancestry at APOE, Fig. 3e) use it; every other local-ancestry model uses
the ROH-region local ancestry already in the TRUROH run. Format: tab-separated, `IID` plus one
`LA_<ancestry>` column per ancestry, each the fraction of the two haplotypes of that ancestry at APOE
(0 / 0.5 / 1), summing to 1 per row:

```
IID     LA_AFR  LA_EUR  LA_NAM
00001   0       0.5     0.5
00002   0.5     0.5     0
```

Build it from the RFMix v2 chr19 output of your TRUROH run:

```bash
python3 TRUROH_AD_Sensitivity/tools/make_apoe_locus_la.py \
  --msp /path/to/rfmix/<prefix>.chr19.msp.tsv --out apoe_locus_LA.tsv
#   --id-map vcf_to_iid.tsv   (columns VCF_ID, IID) if VCF sample names are not the phenotype IIDs
#   --start 45409011 --end 45412650   for GRCh37 (default is GRCh38 APOE, chr19:44,905,754-44,909,393)
```

The script averages the RFMix windows overlapping APOE (weighted by overlap), or uses the nearest window
within 500 kb if none overlaps, and reads ancestry labels from the `.msp` header (AMR/American become NAM).
Check `apoe_locus_local_ancestry.matched_N` in `input_audit.json`: individuals missing from the file are
excluded from E1 only.

## Verify the APOE map before trusting results

`apoe_dosage_map.json` states that codes 0/1/2 mean 0/1/2 ε4 alleles. Confirm with each cohort's data
dictionary. The run stops if any source value is unmapped and warns if the implied ε4 frequency is outside
3–35 % or if cases carry fewer ε4 alleles than controls. Genotype strings (`e3/e4`, `34`, `44`) are
decoded directly. If a cohort records carrier status only, use an `epsilon4_carrier` map; dosage and
genotype models are then reported as not estimable.

## Outputs

| Path | Content |
|---|---|
| `report.html` | Self-contained report: key results, figures + legends, global-ancestry decision, tables, draft methods |
| `interpretation.md`, `summary.json` | Plain-text key results and counts |
| `tables/all_association_tests.tsv` | Every planned test (including not-estimable and original) |
| `tables/all_coefficients.tsv` | Every model term in natural units |
| `tables/effect_contrasts.tsv` | ROH/APOE ORs at modifier values, with observed-range flags |
| `tables/predicted_risk.tsv` | g-computed risks for ROH × ε4 |
| `tables/ancestry_collinearity.tsv`, `tables/carrier_diagnostics.tsv`, `tables/model_plan.tsv` | Diagnostics and the fixed plan |
| `apoe_audit.json`, `apoe_source_value_counts.tsv`, `input_audit.json`, `analysis_manifest.json` | Provenance and audits (catalog, arguments, code hashes) |

## Testing

```bash
bash tests/run_tests.sh                  # 29 unit/integration tests (~1.5 min)
bash tests/e2e/run_e2e.sh /tmp/ad_e2e    # wizard → audit → dry-run → submit (sbatch emulator) → output checks (~1 min)
TRUROH_RUN=/path/completed/output KINSHIP=/path/x.kin0 bash tests/e2e/run_e2e.sh /tmp/ad_e2e   # on a real TRU-ROH run
```

The end-to-end test uses a simulated completed TRU-ROH run (N = 1,000, families, admixture, bare 0/1/2 APOE
codes). It checks the following:

* the audit-only run fits nothing;
* the submission omits empty scheduler options and applies the configured resources;
* the analysis completes, with every planned test reported;
* the primary models converge, APOE decodes as ε4 dosage, and the GLMM and APOE-locus models run;
* the report and all four figure formats are written.

It was also run on the output of the TRUROH_Ancestry end-to-end test (real PLINK/BCFtools calls on a
synthetic cohort).

Unit tests (`tests/run_tests.sh`): 29 tests, including GLMM = logistic ML when there is no relatedness, recovery of
family variance, sandwich formula, KING and GCTA readers, unrelated-set selection, F_ROH interval union; analytic Firth solutions, LR test vs saturated model, score
gradient, exact back-transformation of standardized fits, contrast = coefficient at modifier 0, catalog
rules (no age/sex/PC interactions; hierarchy), APOE decoding/audit/warnings, source-run immutability,
profiles, SLURM submission and spooled worker. `tests/synthetic.py` builds a simulated CH-like run; the
bundled `example_SYNTHETIC_report.html` was produced from it and contains **no real data**.
