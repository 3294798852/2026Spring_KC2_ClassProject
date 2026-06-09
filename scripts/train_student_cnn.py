import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ROOT_DIR, SIMOPA_PATH, STUDENT_CNN_PATH
from src.student_opa import SimOPAStudentCNN, load_frozen_simopa_head


@dataclass(frozen=True)
class OPASample:
    image_path: Path
    mask_path: Path
    label: int


class OPACsvDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        csv_path: Path,
        image_size: int = 256,
        max_samples: int | None = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.samples = self._read_samples(csv_path, max_samples=max_samples)
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path).convert("L")
        image_t = self.rgb_transform(image)
        mask_t = self.mask_transform(mask)
        x = torch.cat([image_t, mask_t], dim=0)
        y = torch.tensor(sample.label, dtype=torch.long)
        return x, y

    def _read_samples(self, csv_path: Path, max_samples: int | None) -> list[OPASample]:
        samples: list[OPASample] = []
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = self._resolve_data_path(row["img_name"])
                mask_path = self._resolve_data_path(row["mask_name"])
                samples.append(
                    OPASample(
                        image_path=image_path,
                        mask_path=mask_path,
                        label=int(row["label"]),
                    )
                )
                if max_samples is not None and len(samples) >= max_samples:
                    break
        if not samples:
            raise ValueError(f"no samples found in {csv_path}")
        return samples

    def _resolve_data_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        candidates = [
            self.dataset_root / path,
            self.dataset_root / Path(*path.parts[1:]) if path.parts and path.parts[0] == "dataset" else None,
            ROOT_DIR / path,
        ]
        for candidate in candidates:
            if candidate is not None and candidate.exists():
                return candidate
        raise FileNotFoundError(f"cannot resolve dataset path: {raw_path}")


def set_trainable_backbone_only(model: SimOPAStudentCNN) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.backbone.parameters():
        param.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total = 0

    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += batch_size

    return total_loss / total, total_correct / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small CNN backbone while keeping SimOPA's prediction head frozen."
    )
    parser.add_argument("--data-root", type=Path, default=ROOT_DIR / "new_OPA")
    parser.add_argument("--train-csv", type=Path, default=ROOT_DIR / "new_OPA" / "train_set.csv")
    parser.add_argument("--val-csv", type=Path, default=ROOT_DIR / "new_OPA" / "test_set.csv")
    parser.add_argument("--simopa-weight", type=Path, default=SIMOPA_PATH)
    parser.add_argument("--output", type=Path, default=STUDENT_CNN_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    train_dataset = OPACsvDataset(
        args.data_root,
        args.train_csv,
        image_size=args.image_size,
        max_samples=args.max_train_samples,
    )
    val_dataset = OPACsvDataset(
        args.data_root,
        args.val_csv,
        image_size=args.image_size,
        max_samples=args.max_val_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SimOPAStudentCNN()
    load_frozen_simopa_head(model, args.simopa_weight)
    set_trainable_backbone_only(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.backbone.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    print(f"device={device}")
    print(f"train_samples={len(train_dataset)} val_samples={len(val_dataset)}")
    print(f"trainable_params={count_trainable_params(model)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model": "SimOPAStudentCNN",
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                args.output,
            )
            print(f"saved_best={args.output}")


if __name__ == "__main__":
    main()
