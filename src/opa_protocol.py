from typing import Protocol

import numpy as np
import torch
import torch.nn as nn


class OPAScorer(Protocol):
    device: torch.device
    model: nn.Module

    def score_batch(self, composites: list[np.ndarray], masks: list[np.ndarray]) -> list[float]:
        ...

