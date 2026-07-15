#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup/infer_example.py — canonical inference example for neural-tape-modeling.

Runs the repo's real flagship inference path (the same model construction +
weight loading + `RNN.predict()` call used by `code/test-model.py`) against a
real, already-trained checkpoint shipped in this mirror's `weights/` folder
(GRU-HS[64]-L[ESR]-DS[ReelToReel_Dataset_MiniPulse100_AKAI_IPS[7.5]_MAXELL]_BEST/best.pth,
~53 KB — no download needed), on a synthetic-but-realistic input signal (an
80 Hz -> 8 kHz exponential chirp, fade-out to zero, 2 s @ 44.1 kHz), and writes
real input/output WAV files to setup/example_output/.

This is a deliberately minimal standalone reproduction of the "MODEL" section
of code/test-model.py — it exercises the actual nonlinear tape network +
trained weights end-to-end, but skips the DATASET / DELAY / NOISE stages of
test-model.py, which require the Zenodo `neural-tape-audio` download (not
fetched here — see setup/README.md Caveats for why and how to extend this).

Usage:
    python setup/infer_example.py [--device {cpu,mps}]

Known upstream bug: RNN.predict()/RNN.warm_start() hard-code
`torch.device("cuda" if torch.cuda.is_available() else "cpu")` internally,
ignoring the device the model/input were actually placed on. On this MPS
machine that means predict() always computes on **cpu** internally regardless
of --device, so keep the model + input on cpu (the default here) to avoid a
"tensors on different devices" RuntimeError. Passing --device mps is left in
only to demonstrate/reproduce that upstream bug on demand.
"""
import argparse
import os
import sys

import numpy as np
import scipy.signal
import soundfile as sf
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))  # model.py uses bare imports (from networks... )

from model import RNN  # noqa: E402

WEIGHT_NAME = "GRU-HS[64]-L[ESR]-DS[ReelToReel_Dataset_MiniPulse100_AKAI_IPS[7.5]_MAXELL]_BEST"
HIDDEN_SIZE = 64  # parsed from "-HS[64]-" in WEIGHT_NAME, per code/utilities/utilities.py:parse_hidden_size
FS = 44100
DURATION_S = 2.0
OUT_DIR = os.path.join(os.path.dirname(__file__), "example_output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu",
                        help="Device to place the model/input on. NOTE: predict()"
                        " always runs its internal buffers on cpu regardless (upstream bug) — "
                        "use mps only to reproduce that failure.")
    args = parser.parse_args()
    device = torch.device(args.device)

    # 1. Real trained checkpoint (already in this mirror, no download needed)
    weight_path = os.path.join(REPO_ROOT, "weights", WEIGHT_NAME, "best.pth")
    assert os.path.exists(weight_path), f"missing checkpoint: {weight_path}"

    model = RNN(input_size=1, hidden_size=HIDDEN_SIZE, output_size=1, skip=False).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded real checkpoint: {weight_path}")

    # 2. Synthetic-but-realistic input signal: exponential chirp 80Hz->8kHz, faded out
    t = np.linspace(0, DURATION_S, int(FS * DURATION_S), endpoint=False)
    chirp = scipy.signal.chirp(t, f0=80, f1=8000, t1=DURATION_S, method="logarithmic")
    fade = np.linspace(1.0, 0.05, len(t))
    audio_in = (0.4 * chirp * fade).astype(np.float32)

    x = torch.from_numpy(audio_in).view(1, 1, -1).to(device)
    print(f"Input: {tuple(x.shape)} ({DURATION_S}s @ {FS}Hz)")

    # 3. Real canonical inference path: RNN.predict() (same call test-model.py makes)
    with torch.inference_mode():
        y = model.predict(x)

    audio_out = y.squeeze().detach().cpu().numpy()
    print(f"Output: {tuple(y.shape)} dtype={y.dtype}")
    print(f"Output stats: min={audio_out.min():.4f} max={audio_out.max():.4f} "
          f"mean={audio_out.mean():.6f} rms={np.sqrt(np.mean(audio_out**2)):.4f}")
    assert audio_out.shape == audio_in.shape, "output length must match input"
    assert np.isfinite(audio_out).all(), "output contains NaN/Inf"

    os.makedirs(OUT_DIR, exist_ok=True)
    in_path = os.path.join(OUT_DIR, "input_chirp.wav")
    out_path = os.path.join(OUT_DIR, "output_tape.wav")
    sf.write(in_path, audio_in, FS)
    sf.write(out_path, audio_out, FS)
    print(f"Wrote {in_path}")
    print(f"Wrote {out_path}")
    print("OK: real-checkpoint tape-model inference completed.")


if __name__ == "__main__":
    main()
