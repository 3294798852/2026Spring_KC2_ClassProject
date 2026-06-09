from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
    """
    Lightweight 4-channel CNN replacement for SimOPA's ResNet-18 backbone.
    """

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


class SimOPAStudentCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = SmallOPAConvBackbone()
        self.avgpool1x1 = nn.AdaptiveAvgPool2d(1)
        self.prediction_head = nn.Linear(512, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(x)
        global_feature = self.avgpool1x1(feature_map).flatten(1)
        return self.prediction_head(global_feature)


def load_frozen_simopa_head(model: SimOPAStudentCNN, simopa_path: Path) -> None:
    state_dict = torch.load(simopa_path, map_location="cpu", weights_only=True)
    model.prediction_head.load_state_dict(
        {
            "weight": state_dict["prediction_head.weight"],
        },
        strict=True,
    )
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
        self.model = SimOPAStudentCNN().to(self.device).eval()
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
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
