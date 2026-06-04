import time
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import COMPRESSED_PATH, STUDENT_PATH
from src.compress import quantize_head
from src.data_synth import generate_synthetic_sample
from src.models import PlacementStudent


def _load(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and obj.get("quantized_head", False):
        model = quantize_head(PlacementStudent())
        model.load_state_dict(obj["state_dict"])
    elif isinstance(obj, dict) and "state_dict" in obj:
        model = PlacementStudent()
        model.load_state_dict(obj["state_dict"])
    else:
        raise ValueError(f"unsupported checkpoint format: {path}")
    model.eval()
    return model


def _make_inputs(n=50):
    xs = []
    for _ in range(n):
        x, _ = generate_synthetic_sample()
        x = np.transpose(x, (2, 0, 1))
        xs.append(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
    return xs


def _bench(model, xs):
    t0 = time.time()
    with torch.no_grad():
        outs = [float(model(x).item()) for x in xs]
    elapsed = (time.time() - t0) * 1000.0
    return elapsed / len(xs), outs


if __name__ == "__main__":
    xs = _make_inputs(n=50)
    raw = _load(STUDENT_PATH)
    raw_ms, raw_out = _bench(raw, xs)
    print(f"raw student avg latency: {raw_ms:.2f} ms")

    if COMPRESSED_PATH.exists():
        cm = _load(COMPRESSED_PATH)
        cm_ms, cm_out = _bench(cm, xs)
        print(f"compressed student avg latency: {cm_ms:.2f} ms")
        corr = np.corrcoef(np.array(raw_out), np.array(cm_out))[0, 1]
        print(f"score correlation(raw vs compressed): {corr:.4f}")
    else:
        print("compressed model not found. run bootstrap_and_compress.py first.")
