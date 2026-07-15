#!/usr/bin/env bash
# ==============================================================================
# run.sh — reproduce the canonical example for neural-tape-modeling.
# Idempotent: safe to re-run. Verified on Apple M5 Max / macOS arm64 / MPS.
#
# Depth: smoke — installs deps, imports code/model.py, instantiates the RNN model,
# and runs a synthetic forward pass on a short audio tensor. No dataset, no
# submodules, no checkpoint downloads.
#
# Usage:
#   ./setup/run.sh          # full setup + run the canonical example
#   SKIP_INSTALL=1 ./setup/run.sh   # skip dependency install, just run
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

# --- 2. run the canonical example ------------------------------------------
# model.py uses bare imports (from networks.unet_1d import ...), so it must run
# with code/ on sys.path — cd into code/ rather than using PYTHONPATH tricks.
(
  cd code
  python -c "
import torch
from model import RNN

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
x = torch.randn(1, 1, 4096, device=device)          # (N_BATCHES, N_CHANNELS, N_SAMPLES)

model = RNN(input_size=1, hidden_size=8, output_size=1, skip=True).to(device)
model.eval()
with torch.no_grad():
    y = model(x)

assert tuple(y.shape) == tuple(x.shape), f'shape mismatch: {y.shape} vs {x.shape}'
print(f'device: {device}')
print(f'RNN forward OK — input: {tuple(x.shape)} output: {tuple(y.shape)} dtype: {y.dtype}')
"
)

echo "OK: neural-tape-modeling canonical example completed."
