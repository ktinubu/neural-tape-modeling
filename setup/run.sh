#!/usr/bin/env bash
# ==============================================================================
# run.sh — reproduce the canonical example for neural-tape-modeling.
# Idempotent: safe to re-run. Verified on Apple M5 Max / macOS arm64 / MPS.
#
# Depth: full — installs deps, loads the repo's real (already-trained) GRU
# checkpoint from weights/, and runs the real code/model.py::RNN.predict()
# inference path (same call code/test-model.py makes) on a synthetic chirp
# signal, producing real input/output WAV files under setup/example_output/.
#
# Usage:
#   ./setup/run.sh                    # full setup + run the canonical example
#   SKIP_INSTALL=1 ./setup/run.sh     # skip dependency install, just run
# ==============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- 1. prerequisites -------------------------------------------------------
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  if [ ! -d .venv ]; then
    uv venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install torch torchaudio numpy scipy soundfile librosa
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# --- 2. run the canonical example -------------------------------------------
# Real checkpoint (weights/GRU-HS[64]-L[ESR]-DS[...AKAI_IPS[7.5]_MAXELL]_BEST/best.pth,
# ~53KB, already committed to this mirror — no download needed) through the
# real RNN.predict() inference path. See setup/infer_example.py for details.
# NOTE: predict()/warm_start() hard-code device="cuda-or-cpu" internally (upstream
# bug) -> always use --device cpu here (default) or it RuntimeErrors on MPS.
python setup/infer_example.py --device cpu

echo "OK: neural-tape-modeling canonical example completed."
