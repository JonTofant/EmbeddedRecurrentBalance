"""Reproducible CPU check of the published policy networks.

Verifies SHA256 checksums, ONNX validity and interfaces, and replays 1024 random
observations (with periodic hidden-state resets) through:

  * GRU_range_C_native.onnx   vs. its TorchScript checkpoint
  * GRU_range_C_stm32ai.onnx  vs. GRU_range_C_native.onnx (outputs and hidden state)

Requirements: numpy, onnx, onnxruntime, torch.   Usage: python validate_models.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent
TOLERANCE = 1e-4
STEPS = 1024

for line in (ROOT / "SHA256SUMS.txt").read_text().splitlines():
    digest, name = line.split()
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name

options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
torch.set_num_threads(1)
sessions = {}
for name in ("GRU_range_C_native.onnx", "GRU_range_C_stm32ai.onnx"):
    onnx.checker.check_model(onnx.load(ROOT / name))
    sessions[name] = ort.InferenceSession(str(ROOT / name), options, providers=["CPUExecutionProvider"])

gru_ts = torch.jit.load(str(ROOT / "torchscript/GRU_range_C.pt"), map_location="cpu").eval()

h_native = np.zeros((1, 1, 64), dtype=np.float32)
h_stm = np.zeros((1, 64), dtype=np.float32)
rng = np.random.default_rng(42)
errors = dict(gru_native_vs_torch_A=0.0, gru_stm32ai_vs_native_A=0.0, gru_hidden=0.0)
for step in range(STEPS):
    if step % 128 == 0:
        h_native.fill(0)
        h_stm.fill(0)
        gru_ts.hidden_state.zero_()
    obs = np.clip(rng.standard_normal((1, 13)), -3, 3).astype(np.float32)
    gru_n, h_native = sessions["GRU_range_C_native.onnx"].run(None, {"obs": obs, "h_in": h_native})
    gru_s, h_stm = sessions["GRU_range_C_stm32ai.onnx"].run(None, {"obs": obs, "h_in": h_stm})
    with torch.no_grad():
        # The exported ONNX graphs include the 2 A * tanh output scaling.
        gru_ref = (2 * torch.tanh(gru_ts(torch.from_numpy(obs)))).numpy()
    for key, a, b in (("gru_native_vs_torch_A", gru_n, gru_ref),
                      ("gru_stm32ai_vs_native_A", gru_s, gru_n), ("gru_hidden", h_stm, h_native.reshape(1, 64))):
        assert np.isfinite(a).all() and np.isfinite(b).all()
        errors[key] = max(errors[key], float(np.max(np.abs(a - b))))
    assert np.max(np.abs(gru_s)) <= 2.00001

passed = max(errors.values()) < TOLERANCE
print(json.dumps({"passed": passed, "tolerance": TOLERANCE, "max_absolute_errors": errors}, indent=2))
raise SystemExit(0 if passed else 1)
