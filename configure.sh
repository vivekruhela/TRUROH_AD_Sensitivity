#!/usr/bin/env bash
# Interactive set-up: writes profiles/<name>.json and config/site.env.  See README.md.
set -euo pipefail
PKG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python3}" "${PKG_ROOT}/tools/configure.py" "$@"
