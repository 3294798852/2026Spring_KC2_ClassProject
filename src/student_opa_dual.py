from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from src.config import STUDENT_DUAL_PATH


class _TinyMaskEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class SimOPAStudentDual(nn.Module):
    """
    Prototype dual-encoder + geometry MLP student.
    Input remains composite+mask to keep integration simple.
    Geometry branch uses bbox-like stats derived from the mask.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        rgb_net = torchvision.models.mobilenet_v3_small(weights=None)
        self.bg_encoder = rgb_net.features
        self.bg_pool = nn.AdaptiveAvgPool2d(1)
        self.mask_encoder = _TinyMaskEncoder()
        self.geom_mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(576 + 32 + 64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.prediction_head = nn.Linear(512, num_classes, bias=False)

    @staticmethod
    def _mask_to_geometry(mask: torch.Tensor) -> torch.Tensor:
        # mask: [N,1,H,W], values in [0,1]
        n, _, h, w = mask.shape
        geom = torch.zeros((n, 5), device=mask.device, dtype=mask.dtype)
        m = (mask > 0.2).float()
        area = m.flatten(1).mean(dim=1)
        ys = torch.linspace(0.0, 1.0, h, device=mask.device, dtype=mask.dtype).view(1, h, 1)
        xs = torch.linspace(0.0, 1.0, w, device=mask.device, dtype=mask.dtype).view(1, 1, w)
        mass = m.sum(dim=(2, 3)) + 1e-6
        cy = (m[:, 0] * ys).sum(dim=(1, 2)) / mass[:, 0]
        cx = (m[:, 0] * xs).sum(dim=(1, 2)) / mass[:, 0]
        h_ratio = (m[:, 0].sum(dim=2) > 0).float().mean(dim=1)
        w_ratio = (m[:, 0].sum(dim=1) > 0).float().mean(dim=1)
        geom[:, 0] = cx
        geom[:, 1] = cy
        geom[:, 2] = w_ratio
        geom[:, 3] = h_ratio
        geom[:, 4] = area
        return geom

    def forward(self, x: torch.Tensor, return_features: bool = False):
        rgb = x[:, :3]
        mask = x[:, 3:4]
        bg_feat = self.bg_pool(self.bg_encoder(rgb)).flatten(1)
        mask_feat = self.mask_encoder(mask)
        geom_feat = self.geom_mlp(self._mask_to_geometry(mask))
        feat = self.fusion(torch.cat([bg_feat, mask_feat, geom_feat], dim=1))
        logits = self.prediction_head(feat)
        if return_features:
            return logits, feat
        return logits


class StudentDualOPAScorer:
    def __init__(
        self,
        device: str = "auto",
        weight_path: Path = STUDENT_DUAL_PATH,
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
                f"student dual weight not found: {self.weight_path}. "
                "Please train it with scripts/train_student_dual.py first."
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
        self.model = SimOPAStudentDual().to(self.device).eval()
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
