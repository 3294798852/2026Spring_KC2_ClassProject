import copy
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.utils.prune as prune

from src.config import COMPRESSED_PATH, STUDENT_PATH
from src.models import PlacementStudent


def prune_student(model: PlacementStudent, amount: float = 0.15) -> PlacementStudent:
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model


def quantize_head(model: PlacementStudent) -> PlacementStudent:
    model_q = copy.deepcopy(model).cpu()
    model_q.head = torch.quantization.quantize_dynamic(
        model_q.head, {torch.nn.Linear}, dtype=torch.qint8
    )
    return model_q


def compress_student_model(
    src_path: Path = STUDENT_PATH, dst_path: Path = COMPRESSED_PATH
) -> Tuple[float, float]:
    model = PlacementStudent()
    ckpt = torch.load(src_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    model = prune_student(model, amount=0.15)
    model_q = quantize_head(model)

    torch.save({"state_dict": model_q.state_dict(), "quantized_head": True}, dst_path)
    src_mb = src_path.stat().st_size / (1024 * 1024)
    dst_mb = dst_path.stat().st_size / (1024 * 1024)
    return src_mb, dst_mb
