from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from src.config import STUDENT_CNN_PATH


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SmallOPAConvBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(4, 24, stride=2),
            ConvBNReLU(24, 48, stride=2),
            ConvBNReLU(48, 96, stride=2),
            ConvBNReLU(96, 160, stride=2),
            ConvBNReLU(160, 256, stride=2),
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class LegacySimOPAStudentCNN(nn.Module):
    """
    Old student architecture kept for loading legacy checkpoints.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = SmallOPAConvBackbone()
        self.avgpool1x1 = nn.AdaptiveAvgPool2d(1)
        self.prediction_head = nn.Linear(512, num_classes, bias=False)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        fmap = self.backbone(x)
        feat = self.avgpool1x1(fmap).flatten(1)
        logits = self.prediction_head(feat)
        if return_features:
            return logits, feat
        return logits


class SimOPAStudentCNN(nn.Module):
    """
    MobileNetV3-Small student with 4-channel input (RGB+mask).
    Kept class name for backward compatibility with existing scripts.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        net = torchvision.models.mobilenet_v3_small(weights=None)
        first_conv = net.features[0][0]
        net.features[0][0] = nn.Conv2d(
            4,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        self.backbone = net.features
        self.avgpool1x1 = nn.AdaptiveAvgPool2d(1)
        feat_dim = 576
        self.feature_proj = nn.Linear(feat_dim, 512, bias=False)
        self.prediction_head = nn.Linear(512, num_classes, bias=False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(x)
        pooled = self.avgpool1x1(fmap).flatten(1)
        return self.feature_proj(pooled)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.forward_features(x)
        logits = self.prediction_head(feat)
        if return_features:
            return logits, feat
        return logits


def load_frozen_simopa_head(model: SimOPAStudentCNN, simopa_path: Path) -> None:
    """
    Legacy helper retained for compatibility.
    If teacher head shape mismatches (expected for MobileNet student), keep
    student head trainable and skip strict loading.
    """
    state_dict = torch.load(simopa_path, map_location="cpu", weights_only=True)
    teacher_w = state_dict.get("prediction_head.weight")
    if teacher_w is not None and tuple(teacher_w.shape) == tuple(model.prediction_head.weight.shape):
        model.prediction_head.load_state_dict({"weight": teacher_w}, strict=True)
        for param in model.prediction_head.parameters():
            param.requires_grad = False


class StudentOPAScorer:
    def __init__(
        self,
        device: str = "auto",
        weight_path: Path = STUDENT_CNN_PATH,
        score_temperature: float = 1.2,
    ) -> None:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.weight_path = weight_path
        if not self.weight_path.exists():
            raise FileNotFoundError(
                f"student CNN weight not found: {self.weight_path}. "
                "Please train it with scripts/train_student_cnn.py first."
            )

        checkpoint = torch.load(self.weight_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        def _normalize_state_dict(sd: dict) -> dict:
            out = {}
            for k, v in sd.items():
                nk = k
                if nk.startswith("_orig_mod."):
                    nk = nk[len("_orig_mod.") :]
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                out[nk] = v
            return out

        state_dict = _normalize_state_dict(state_dict)

        def _looks_like_legacy(sd: dict) -> bool:
            return any(k.startswith("backbone.features.") for k in sd.keys())

        if _looks_like_legacy(state_dict):
            self.model = LegacySimOPAStudentCNN().to(self.device).eval()
            self.model.load_state_dict(state_dict, strict=True)
        else:
            self.model = SimOPAStudentCNN().to(self.device).eval()
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                # Final fallback for unexpected legacy naming.
                self.model = LegacySimOPAStudentCNN().to(self.device).eval()
                self.model.load_state_dict(state_dict, strict=True)
        self.temperature = max(0.1, float(score_temperature))
        image_size = int(checkpoint.get("image_size", 256))
        self.transformer = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
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
        scores = torch.clamp(scores, min=1e-6, max=1 - 1e-6)
        return scores.detach().cpu().numpy().astype(float).tolist()
