#!/usr/bin/env bash
# Install ParamRF with the full test environment used by CI.
#
# CI and local development both run this, so a test that is skipped locally for a
# missing optional backend is also run locally. Needs gfortran and MPI for PolyChord.
#
# Usage: scripts/install-test-deps.sh [python]   (default: python)
set -euo pipefail

PYTHON="${1:-python}"

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -e ".[tests]"

# The package must import with only its declared dependencies, before any override.
"$PYTHON" -I -c "import pmrf"

# Backend overrides. Pinned so CI behaviour cannot change underneath a commit.
"$PYTHON" -m pip uninstall -y distreqx
"$PYTHON" -m pip install "git+https://github.com/gvcallen/distreqx.git@044ea5caf3bb2b92de54b82ea1edc485c657c906"
"$PYTHON" -m pip install "git+https://github.com/PolyChord/PolyChordLite.git@370f6af5d59a4d3af2ed3333acfba152fdb7a4cf"
"$PYTHON" -m pip install "git+https://github.com/handley-lab/blackjax.git@v0.1.0-beta"
