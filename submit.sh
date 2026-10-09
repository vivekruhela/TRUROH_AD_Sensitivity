#!/usr/bin/env bash
# Submit the association analysis as one SLURM job.
#
#   bash submit.sh --profile my_cohort [analysis options]          # profile written by configure.sh
#   bash submit.sh --config /abs/TRUROH_Ancestry/config/x.json --apoe-map apoe_dosage_map.json
#   bash submit.sh --dry-run --profile my_cohort                   # print the sbatch command only
#
# Scheduler settings come from config/site.env (written by configure.sh; another file via TRUROH_AD_SITE_ENV)
# or the environment:
#   TRUROH_AD_PARTITION, TRUROH_AD_ACCOUNT, TRUROH_AD_CONSTRAINT  (empty = option omitted)
#   TRUROH_AD_CPUS (4), TRUROH_AD_MEMORY (16G), TRUROH_AD_TIME (1-00:00:00), TRUROH_AD_LOG_DIR (<package>/logs)
#   TRUROH_AD_DEPENDENCY  optional SLURM dependency, e.g. afterok:12345 to start after a TRU-ROH finalizer
# The v2.0.0 names TRUROH_CONSTRAINT/TRUROH_CPUS/TRUROH_MEMORY/TRUROH_TIME are still honoured.
set -euo pipefail
PKG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRUROH_AD_SITE_ENV="${TRUROH_AD_SITE_ENV:-${PKG_ROOT}/config/site.env}"
if [[ -r "${TRUROH_AD_SITE_ENV}" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${TRUROH_AD_SITE_ENV}"
  set -u
fi
DRY_RUN=0
ARGS=()
for a in "$@"; do
  if [[ "$a" == "--dry-run" ]]; then DRY_RUN=1; else ARGS+=("$a"); fi
done
if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "Usage: bash submit.sh [--dry-run] --profile NAME|path.json [options]" >&2
  echo "   or: bash submit.sh [--dry-run] --config /abs/cohort.json --apoe-map MAP [options]" >&2
  echo "Run 'bash configure.sh' first to create a profile and scheduler settings." >&2
  exit 2
fi

partition="${TRUROH_AD_PARTITION:-}"
account="${TRUROH_AD_ACCOUNT:-}"
constraint="${TRUROH_AD_CONSTRAINT:-${TRUROH_CONSTRAINT:-}}"
cpus="${TRUROH_AD_CPUS:-${TRUROH_CPUS:-4}}"
memory="${TRUROH_AD_MEMORY:-${TRUROH_MEMORY:-16G}}"
walltime="${TRUROH_AD_TIME:-${TRUROH_TIME:-1-00:00:00}}"
log_dir="${TRUROH_AD_LOG_DIR:-${PKG_ROOT}/logs}"

export TRUROH_AD_SITE_ENV
cmd=(sbatch --parsable --job-name="${TRUROH_AD_JOB_NAME:-TRUROH_AD_assoc}"
     --cpus-per-task="${cpus}" --mem="${memory}" --time="${walltime}")
[[ -n "${partition}" ]] && cmd+=(--partition="${partition}")
[[ -n "${account}" ]] && cmd+=(--account="${account}")
[[ -n "${constraint}" ]] && cmd+=(--constraint="${constraint}")
[[ -n "${TRUROH_AD_DEPENDENCY:-}" ]] && cmd+=(--dependency="${TRUROH_AD_DEPENDENCY}")
cmd+=(--chdir="$PWD" --output="${log_dir}/truroh_ad_%j.out" --error="${log_dir}/truroh_ad_%j.err"
      "${PKG_ROOT}/slurm/job.sbatch" "${PKG_ROOT}" "${ARGS[@]}")

# Check arguments, input files and the completed TRU-ROH run before anything is queued.
check=("${PYTHON_BIN:-python3}" "${PKG_ROOT}/tools/check_args.py")
[[ -n "${TRUROH_AD_DEPENDENCY:-}" ]] && check+=(--allow-pending-run)
"${check[@]}" "${ARGS[@]}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%q ' "${cmd[@]}"; echo
  exit 0
fi
mkdir -p "${log_dir}"
# sbatch errors are printed verbatim.
exec "${cmd[@]}"
