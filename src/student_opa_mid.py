from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from src.config import STUDENT_MID_PATH


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class SimOPAStudentMid(nn.Module):
    """
    Mid-size student model (~6M params): ResNet18-style 4ch network with
    width-scaled channels [48, 96, 192, 384].
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.inplanes = 48
        self.stem = nn.Sequential(
            nn.Conv2d(4, 48, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(48, blocks=2, stride=1)
        self.layer2 = self._make_layer(96, blocks=2, stride=2)
        self.layer3 = self._make_layer(192, blocks=2, stride=2)
        self.layer4 = self._make_layer(384, blocks=2, stride=2)
        self.avgpool1x1 = nn.AdaptiveAvgPool2d(1)
        self.feature_proj = nn.Linear(384, 512, bias=False)
        self.prediction_head = nn.Linear(512, num_classes, bias=False)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.inplanes, planes, stride=stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool1x1(x).flatten(1)
        return self.feature_proj(x)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.forward_features(x)
        logits = self.prediction_head(feat)
        if return_features:
            return logits, feat
        return logits


class StudentMidOPAScorer:
    def __init__(
        self,
        device: str = "auto",
        weight_path: Path = STUDENT_MID_PATH,
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
                f"student mid weight not found: {self.weight_path}. "
                "Please train it with scripts/train_student_mid.py first."
            )
        checkpoint = torch.load(self.weight_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        normalized = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("_orig_mod."):
                nk = nk[len("_orig_mod.") :]
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            normalized[nk] = v

        self.model = SimOPAStudentMid().to(self.device).eval()
        self.model.load_state_dict(normalized, strict=True)
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
