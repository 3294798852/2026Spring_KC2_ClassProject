import torch

from src.reference_opa import ReferenceOPAScorer
from src.student_opa import StudentOPAScorer


STUDENT_BACKEND = "Student CNN"
REFERENCE_BACKEND = "原始 SimOPA"
BACKENDS = [STUDENT_BACKEND, REFERENCE_BACKEND]


def select_opa_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_opa_scorer(model_backend: str, device: str = "auto"):
    selected_device = select_opa_device(device)
    if model_backend == REFERENCE_BACKEND:
        return ReferenceOPAScorer(device=str(selected_device))
    if model_backend == STUDENT_BACKEND:
        return StudentOPAScorer(device=str(selected_device))
    raise ValueError(f"unknown OPA backend: {model_backend}")
