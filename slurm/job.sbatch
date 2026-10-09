#!/usr/bin/env bash
set -euo pipefail
# SLURM copies this worker to its spool directory, so the package path is passed explicitly.
if [[ $# -lt 1 ]]; then
  echo "Missing package path" >&2
  exit 2
fi
PKG_ROOT="$1"
shift
# Site settings written by configure.sh (modules / conda environment), if present.
TRUROH_AD_SITE_ENV="${TRUROH_AD_SITE_ENV:-${PKG_ROOT}/config/site.env}"
if [[ -r "${TRUROH_AD_SITE_ENV}" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${TRUROH_AD_SITE_ENV}"
  set -u
fi
echo "[TRUROH-AD] host=$(hostname) job=${SLURM_JOB_ID:-local} cpus=${SLURM_CPUS_PER_TASK:-1} python=$(command -v "${PYTHON_BIN:-python3}")"
exec bash "${PKG_ROOT}/run.sh" "$@"
