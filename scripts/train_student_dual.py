import argparse
import contextlib
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_student_cnn import OPACsvDataset, SimOPATeacherWithFeatures, kd_loss, rank_distill_loss, select_device
from src.config import ROOT_DIR, SIMOPA_PATH, STUDENT_DUAL_PATH
from src.student_opa_dual import SimOPAStudentDual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dual-encoder + geometry OPA student.")
    parser.add_argument("--data-root", type=Path, default=ROOT_DIR / "new_OPA")
    parser.add_argument("--train-csv", type=Path, default=ROOT_DIR / "new_OPA" / "train_set.csv")
    parser.add_argument("--val-csv", type=Path, default=ROOT_DIR / "new_OPA" / "test_set.csv")
    parser.add_argument("--simopa-weight", type=Path, default=SIMOPA_PATH)
    parser.add_argument("--output", type=Path, default=STUDENT_DUAL_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha-kd", type=float, default=0.7)
    parser.add_argument("--beta-feat", type=float, default=0.2)
    parser.add_argument("--gamma-rank", type=float, default=0.1)
    parser.add_argument("--disable-amp", action="store_true", help="disable mixed precision on CUDA")
    parser.add_argument("--channels-last", action="store_true", help="use channels_last tensor memory format")
    parser.add_argument("--compile-model", action="store_true", help="use torch.compile for student/teacher")
    parser.add_argument("--compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=ROOT_DIR / "logs")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def _setup_run_logging(log_dir: Path, run_name: str | None, config: dict) -> tuple[Path, Path]:
    run_id = run_name or datetime.now().strftime("student_dual_%Y%m%d_%H%M%S")
    run_dir = log_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_csv = run_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "epoch_time_sec",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "best_val_acc",
                "lr",
            ],
        )
        writer.writeheader()
    return run_dir, metrics_csv


def _append_metrics_row(metrics_csv: Path, row: dict) -> None:
    with metrics_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def run_epoch(
    model: SimOPAStudentDual,
    teacher: SimOPATeacherWithFeatures,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    temperature: float,
    alpha_kd: float,
    beta_feat: float,
    gamma_rank: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_channels_last: bool = False,
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
            if use_channels_last and x.ndim == 4:
                x = x.contiguous(memory_format=torch.channels_last)
            amp_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with amp_ctx:
                with torch.no_grad():
                    t_logits, t_feat = teacher(x)
                    # Avoid cudagraph output buffer reuse issues under torch.compile.
                    t_logits = t_logits.detach().clone()
                    t_feat = t_feat.detach().clone()
                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                s_logits, s_feat = model(x, return_features=True)
                loss_ce = criterion(s_logits, y)
                loss_kd = kd_loss(s_logits, t_logits, temperature=temperature)
                loss_feat = F.smooth_l1_loss(s_feat, t_feat)
                loss_rank = rank_distill_loss(
                    torch.softmax(s_logits, dim=-1)[:, 1],
                    torch.softmax(t_logits, dim=-1)[:, 1],
                )
                loss = loss_ce + alpha_kd * loss_kd + beta_feat * loss_feat + gamma_rank * loss_rank
            if is_train:
                if scaler is not None and use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            b = y.size(0)
            total += b
            total_loss += loss.item() * b
            total_correct += (s_logits.argmax(dim=1) == y).sum().item()
    return total_loss / max(1, total), total_correct / max(1, total)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    use_amp = (device.type == "cuda") and (not args.disable_amp)
    amp_dtype = torch.float16
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
        loader_kwargs["persistent_workers"] = not args.no_persistent_workers

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
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    teacher = SimOPATeacherWithFeatures(args.simopa_weight).to(device)
    model = SimOPAStudentDual().to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        teacher = teacher.to(memory_format=torch.channels_last)
    if args.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"torch.compile enabled for student: mode={args.compile_mode}")
        except Exception as exc:
            print(f"torch.compile unavailable, fallback eager: {exc}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    else:
        scaler = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_dir, metrics_csv = _setup_run_logging(
        log_dir=args.log_dir,
        run_name=args.run_name,
        config={
            "script": "train_student_dual.py",
            "device": str(device),
            "output": str(args.output),
            "args": vars(args),
        },
    )
    best_val_acc = -1.0
    print(
        f"runtime: amp={use_amp} channels_last={bool(args.channels_last)} "
        f"compile={bool(args.compile_model)} workers={args.num_workers}"
    )
    print(f"log_dir={run_dir}")
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_loss, train_acc = run_epoch(
            model, teacher, train_loader, criterion, device,
            temperature=args.temperature,
            alpha_kd=args.alpha_kd,
            beta_feat=args.beta_feat,
            gamma_rank=args.gamma_rank,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            use_channels_last=bool(args.channels_last),
            optimizer=optimizer,
        )
        val_loss, val_acc = run_epoch(
            model, teacher, val_loader, criterion, device,
            temperature=args.temperature,
            alpha_kd=args.alpha_kd,
            beta_feat=args.beta_feat,
            gamma_rank=args.gamma_rank,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=None,
            use_channels_last=bool(args.channels_last),
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(
                {
                    "model": "SimOPAStudentDual",
                    "model_state_dict": model_to_save.state_dict(),
                    "image_size": args.image_size,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                args.output,
            )
            print(f"saved_best={args.output}")
        epoch_time = time.time() - t_epoch
        lr_now = float(optimizer.param_groups[0]["lr"])
        _append_metrics_row(
            metrics_csv,
            {
                "epoch": int(epoch),
                "epoch_time_sec": f"{epoch_time:.3f}",
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
                "best_val_acc": f"{best_val_acc:.6f}",
                "lr": f"{lr_now:.8f}",
            },
        )

    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_val_acc": float(best_val_acc),
                "output_weight": str(args.output),
                "metrics_csv": str(metrics_csv),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
