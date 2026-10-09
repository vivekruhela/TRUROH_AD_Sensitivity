#!/usr/bin/env bash
# Run on an interactive/compute node: bash run.sh --profile ch [--audit-only]
set -euo pipefail
PKG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLBACKEND=Agg
exec "${PYTHON_BIN:-python3}" "${PKG_ROOT}/roh_assoc.py" "$@"
