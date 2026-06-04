from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.config import STUDENT_PATH, TEACHER_PATH
from src.data_synth import generate_synthetic_sample
from src.models import PlacementStudent, PlacementTeacher


class SyntheticPlacementDataset(Dataset):
    def __init__(self, size: int = 8000) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        x, y = generate_synthetic_sample()
        x = np.transpose(x, (2, 0, 1))
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)


@dataclass
class TrainResult:
    teacher_loss: float
    student_loss: float
    student_val_mae: float
    student_val_corr: float


def train_teacher_student(
    device: str = "cpu",
    epochs_teacher: int = 3,
    epochs_student: int = 5,
    batch_size: int = 32,
    train_size: int = 3000,
    val_size: int = 600,
    seed: int = 42,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_set = SyntheticPlacementDataset(size=train_size)
    val_set = SyntheticPlacementDataset(size=val_size)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    teacher = PlacementTeacher().to(device)
    student = PlacementStudent().to(device)
    bce = nn.BCELoss()

    teacher_opt = optim.Adam(teacher.parameters(), lr=7e-4, weight_decay=1e-4)
    teacher_last = 0.0
    for _ in range(epochs_teacher):
        teacher.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = teacher(x)
            loss = bce(pred, y)
            teacher_opt.zero_grad()
            loss.backward()
            teacher_opt.step()
            total += float(loss.item())
        teacher_last = total / len(loader)

    torch.save({"state_dict": teacher.state_dict()}, TEACHER_PATH)

    student_opt = optim.Adam(student.parameters(), lr=6e-4, weight_decay=1e-4)
    mse = nn.MSELoss()
    alpha = 0.65
    student_last = 0.0
    for _ in range(epochs_student):
        student.train()
        teacher.eval()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t_pred = teacher(x)
            s_pred = student(x)
            loss = alpha * bce(s_pred, y) + (1 - alpha) * mse(s_pred, t_pred)
            student_opt.zero_grad()
            loss.backward()
            student_opt.step()
            total += float(loss.item())
        student_last = total / len(loader)

    student.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = student(x)
            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    y_true_np = np.concatenate(y_true, axis=0).reshape(-1)
    y_pred_np = np.concatenate(y_pred, axis=0).reshape(-1)
    mae = float(np.mean(np.abs(y_true_np - y_pred_np)))
    corr = float(np.corrcoef(y_true_np, y_pred_np)[0, 1]) if np.std(y_pred_np) > 1e-8 else 0.0

    torch.save({"state_dict": student.state_dict(), "val_mae": mae, "val_corr": corr}, STUDENT_PATH)
    return TrainResult(
        teacher_loss=teacher_last, student_loss=student_last, student_val_mae=mae, student_val_corr=corr
    )


def load_student(device: str = "cpu", model_path: Optional[str] = None) -> PlacementStudent:
    model = PlacementStudent().to(device)
    path = STUDENT_PATH if model_path is None else model_path
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
