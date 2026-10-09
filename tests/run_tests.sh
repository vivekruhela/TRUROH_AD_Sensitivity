#!/usr/bin/env bash
set -euo pipefail
TEST_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd -- "$TEST_ROOT"
exec "${PYTHON_BIN:-python3}" -m unittest discover -s tests -p 'test_*.py' -v
