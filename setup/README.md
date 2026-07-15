# neural-tape-modeling — setup & canonical example (khaledt-dev)

> Private mirror of `01tot10/neural-tape-modeling`. This folder documents **exactly** how the
> canonical example was brought up on this machine, reproducibly via `./setup/run.sh`.

## What this repo is / what the canonical example does
Research code (DAFx23 paper) for neural modeling of magnetic tape recorders in PyTorch. It
ships several model architectures in `code/model.py` — a GRU `RNN`, a `DiffDelRNN` (GRU +
differentiable delay line), and a `DiffusionGenerator` (UNet1D, needs a trained checkpoint).
The canonical example chosen here is instantiating the flagship `RNN` model (as used by
`code/train.py`/`code/test-model.py`) and running a synthetic forward pass on a short audio
tensor — this is the smallest self-contained slice that exercises the real network code
without requiring the Zenodo dataset, submodule checkouts, or trained weights.

## System it was verified on
- Apple M5 Max, macOS arm64 (Darwin). Python 3.11.15 via uv 0.11.28. Torch backend: **MPS** (no CUDA).
- Depth reached: **smoke**
- Wall-clock to reproduce: **~1 min** (deps were already cached by uv from prior repos in this sweep)
  · Extra disk used: **~812 MB** (`.venv`, mostly torch/torchaudio wheels)

## Prerequisites
- `uv` (0.11+), `python3.11` — both already present on this machine, no brew installs needed.
- No system tools (cmake/ninja/faust) needed — pure Python.
- No submodules initialized. The upstream repo's `.gitmodules` lists 6 submodules
  (`AnalogTapeModel`, `micro_tcn`, `Automated_GuitarAmpModelling`, `GreyBoxDRC`, `auraloss`,
  `edm`) used for training/loss functions and target generation — **not required** for the
  smoke-depth import + forward pass, since `code/model.py`'s `RNN`/`DiffDelRNN` classes only
  depend on `numpy`, `torch`, `torchaudio`, and the repo's own `code/networks/unet_1d.py` +
  `code/utilities/utilities.py` (which in turn need `scipy`, `soundfile`, `librosa`).
- No dataset download (upstream README points at a Zenodo tape-audio dataset — skipped at
  this depth) and no checkpoint download (only needed for `DiffusionGenerator`, not exercised).

## Exact steps performed (copy-paste reproducible)
```bash
cd /Users/km/dev/neural-tape-modeling

# 1. venv + deps (upstream's environment.yaml targets conda/mamba + CUDA; we use uv + CPU/MPS
#    torch instead, installing only what code/model.py's RNN path actually imports)
uv venv .venv
source .venv/bin/activate
uv pip install torch torchaudio numpy scipy soundfile librosa

# 2. smoke test: import the real model module and run a forward pass on synthetic audio
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
print('RNN forward OK — input:', tuple(x.shape), 'output:', tuple(y.shape), 'dtype:', y.dtype)
"
```

Note: `model.py` uses bare imports (`from networks.unet_1d import UNet1D`, `from
utilities.utilities import nextpow2`), i.e. it assumes `code/` itself is on `sys.path` — so
the smoke script must run with cwd `code/` (or `code/` prepended to `PYTHONPATH`), not from
the repo root.

## Expected output
```
$ uv pip install torch torchaudio numpy scipy soundfile librosa
Resolved 36 packages in 4ms
Installed 36 packages in 245ms
 + torch==2.13.0
 + torchaudio==2.11.0
 ... (34 more, see setup/requirements.lock.txt)

$ python -c "..."
device: mps
RNN forward OK — input: (1, 1, 4096) output: (1, 1, 4096) dtype: torch.float32
```

## Caveats / boundary
- **Smoke depth only**: no training, no dataset, no checkpoint download. The real
  training/eval pipelines (`code/train.py`, `code/scripts/*.sh`) need the Zenodo tape-audio
  dataset symlinked at `audio/` and the 6 git submodules checked out — none of that was
  attempted here.
- **`RNN.predict()` / `RNN.warm_start()` are MPS-broken (upstream bug, not something we
  patched)**: they hard-code `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
  internally, ignoring the device the model was actually moved to. On this MPS machine calling
  `model.predict(x)` with a model on `mps` raises `RuntimeError: Input and parameter tensors
  are not at the same device, found input tensor at cpu and parameter tensor at mps:0` — the
  same bug exists in `DiffDelRNN`. Plain `model(x)` (the `forward()` we smoke-test) is
  unaffected because it doesn't call `.to(device)` internally. Verified: `predict()` works
  fine when the model/tensor are kept on `cpu` instead of `mps`.
- `DiffusionGenerator` (the UNet1D diffusion model) was not smoke-tested — its constructor
  requires `args.network.checkpoint`, a trained weights file not present in this mirror.
- `environment.yaml` targets conda/mamba + `pytorch-cuda=11.7`; we used `uv` + CPU/MPS wheels
  instead (no CUDA on this machine), so package versions differ from upstream's pinned conda
  env (`environment.yaml` doesn't pin versions anyway).

## Troubleshooting
- No errors were hit during install (`uv pip install` resolved cleanly against the cached
  wheel cache from earlier repos in this sweep — first-time install of torch/torchaudio would
  take longer / use more network).
- If you see the `cuda`/`mps` device-mismatch `RuntimeError` above, it's the upstream
  `predict()`/`warm_start()` device bug described in Caveats — use `forward()` directly, or
  force `cpu`.
