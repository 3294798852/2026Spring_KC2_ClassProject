import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from src.config import SIMOPA_PATH


def _select_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SimOPAResNet(nn.Module):
    """
    SimOPA model from BCMI/libcom (OPA score).
    Architecture matches the released SimOPA checkpoint.
    """

    def __init__(self) -> None:
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool1x1 = nn.AdaptiveAvgPool2d(1)
        self.prediction_head = nn.Linear(512, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(x)
        global_feature = self.avgpool1x1(feature_map).flatten(1)
        return self.prediction_head(global_feature)


def _download_simopa_weight(dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return

    # Try HuggingFace first, then fallback to ModelScope mirror used by libcom.
    try:
        from huggingface_hub import hf_hub_download

        file_path = hf_hub_download(
            repo_id="BCMIZB/Libcom_pretrained_models",
            filename="SimOPA.pth",
            cache_dir=str(dst_path.parent),
        )
        shutil.copyfile(file_path, dst_path)
        return
    except Exception:
        pass

    from modelscope.hub.file_download import model_file_download

    file_path = model_file_download(
        model_id="yujieouo/Libcom_pretrained_models",
        file_path="SimOPA.pth",
        cache_dir=str(dst_path.parent),
        revision="master",
    )
    shutil.copyfile(file_path, dst_path)


def ensure_simopa_weight(weight_path: Path = SIMOPA_PATH) -> Path:
    if not weight_path.exists():
        try:
            _download_simopa_weight(weight_path)
        except Exception as exc:
            raise RuntimeError(
                f"failed to prepare SimOPA weight at {weight_path}. "
                "Please check network or manually place SimOPA.pth into models/."
            ) from exc
    return weight_path


class ReferenceOPAScorer:
    def __init__(
        self,
        device: str = "auto",
        weight_path: Optional[Path] = None,
        score_temperature: float = 1.2,
    ) -> None:
        self.device = _select_device(device)
        self.weight_path = ensure_simopa_weight(SIMOPA_PATH if weight_path is None else weight_path)
        self.model = SimOPAResNet().to(self.device).eval()
        state_dict = torch.load(self.weight_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        self.temperature = max(0.1, float(score_temperature))
        self.transformer = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

    def _preprocess(self, composite_image: np.ndarray, composite_mask: np.ndarray) -> torch.Tensor:
        img = Image.fromarray(np.uint8(np.clip(composite_image, 0.0, 1.0) * 255), mode="RGB")
        mask = Image.fromarray(np.uint8(np.clip(composite_mask[..., 0], 0.0, 1.0) * 255), mode="L")
        img_t = self.transformer(img)
        mask_t = self.transformer(mask)
        return torch.cat([img_t, mask_t], dim=0)

    @torch.no_grad()
    def score_batch(self, composites: list[np.ndarray], masks: list[np.ndarray]) -> list[float]:
        inputs = [self._preprocess(comp, mask) for comp, mask in zip(composites, masks)]
        batch = torch.stack(inputs, dim=0).to(self.device)
        logits = self.model(batch) / self.temperature
        scores = torch.softmax(logits, dim=-1)[:, 1]
        # Keep scores in open interval to reduce apparent hard 0/1 saturation in UI.
        scores = torch.clamp(scores, min=1e-6, max=1 - 1e-6)
        return scores.detach().cpu().numpy().astype(float).tolist()
