#!/usr/bin/env bash
# Locate official SONIC checkpoints; optional release-only conversion.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m orcs.cli.sonic_checkpoint "$@"
