#!/usr/bin/env bash
# End-to-end test of the user path: configure.sh -> run.sh --audit-only -> submit.sh --dry-run ->
# submit.sh through a synchronous sbatch emulator -> output checks.
#
#   bash tests/e2e/run_e2e.sh [WORKDIR]                       # simulated completed TRU-ROH run (tests/synthetic.py)
#   TRUROH_RUN=/path/to/completed/output KINSHIP=/path/x.kin0 bash tests/e2e/run_e2e.sh [WORKDIR]
#                                                             # any completed TRU-ROH run, e.g. from TRUROH_Ancestry's e2e
set -euo pipefail
PKG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${1:-$(mktemp -d "${TMPDIR:-/tmp}/truroh_ad_e2e.XXXXXX")}"
WORK="$(mkdir -p "${WORK}" && cd "${WORK}" && pwd)"
PY="${PYTHON_BIN:-python3}"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[e2e] work directory: ${WORK}"

if [[ -z "${TRUROH_RUN:-}" ]]; then
  echo "[e2e] 1/5 simulated completed TRU-ROH run (N=1000, 1 high-confidence region)"
  "${PY}" - "${WORK}/sim" "${PKG_ROOT}" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[2] + "/tests")
from synthetic import make_run
run, *_ = make_run(sys.argv[1], n=1000, n_carriers=70, seed=11)
print(run)
PYEOF
  TRUROH_RUN="${WORK}/sim/completed run"
  KINSHIP="${KINSHIP:-${WORK}/sim/cohort.kin0}"
  APOE_LA="${WORK}/sim/apoe_locus_la.tsv"
else
  echo "[e2e] 1/5 using completed TRU-ROH run ${TRUROH_RUN}"
  APOE_LA=""
fi

echo "[e2e] 2/5 configuration wizard (non-interactive answers)"
"${PY}" - "${WORK}/answers.json" "${TRUROH_RUN}" "${KINSHIP:-}" "${APOE_LA}" "${WORK}/results" <<'PYEOF'
import json, sys
out, run, kin, la, res = sys.argv[1:]
json.dump({"run": run, "apoe_source": "", "apoe_column": "APOE_reported", "apoe_map": "@pkg/apoe_dosage_map.json",
           "kinship": kin, "primary": "firth", "inbreeding": "", "apoe_locus_la": la, "min_carriers": 3,
           "out_parent": res, "out_name": "E2E_AD", "profile_name": "e2e",
           "slurm.partition": "", "slurm.account": "", "slurm.constraint": "", "slurm.cpus": 2, "slurm.memory": "4G",
           "slurm.time": "01:00:00", "site.modules": "", "site.conda": "", "site.python": sys.executable}, open(out, "w"))
PYEOF
bash "${PKG_ROOT}/configure.sh" --answers "${WORK}/answers.json" --profile-dir "${WORK}/profiles" --site-env "${WORK}/site.env" > "${WORK}/configure.log"
export TRUROH_AD_SITE_ENV="${WORK}/site.env"
PROFILE="${WORK}/profiles/e2e.json"

echo "[e2e] 3/5 audit-only run"
bash "${PKG_ROOT}/run.sh" --profile "${PROFILE}" --audit-only --out-dir "${WORK}/audit" > "${WORK}/audit.log" 2>&1

echo "[e2e] 4/5 dry-run, then SLURM submission through the sbatch emulator"
export FAKE_SLURM_STATE="${WORK}/fake_slurm"
PATH="${PKG_ROOT}/tests/e2e/fake_slurm:${PATH}" bash "${PKG_ROOT}/submit.sh" --dry-run --profile "${PROFILE}" --prediction-draws 200 > "${WORK}/dryrun.log"
test ! -e "${FAKE_SLURM_STATE}/sbatch_calls.jsonl"
(cd "${WORK}" && PATH="${PKG_ROOT}/tests/e2e/fake_slurm:${PATH}" TRUROH_AD_LOG_DIR="${WORK}/logs" \
   bash "${PKG_ROOT}/submit.sh" --profile "${PROFILE}" --prediction-draws 200 > "${WORK}/submit.log")

echo "[e2e] 5/5 output checks"
"${PY}" "${PKG_ROOT}/tests/e2e/check_outputs.py" --results "${WORK}/results" --audit "${WORK}/audit" \
  --sbatch-log "${FAKE_SLURM_STATE}/sbatch_calls.jsonl" --kinship "${KINSHIP:-}" --apoe-la "${APOE_LA}"
echo "[e2e] PASSED  (outputs in ${WORK})"
