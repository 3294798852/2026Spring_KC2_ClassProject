import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_student_cnn import (
    OPACsvDataset,
    SimOPATeacherWithFeatures,
    _append_metrics_row,
    _setup_run_logging,
    feature_distill_loss,
    kd_loss,
    prepare_train_val_csv,
    rank_distill_loss,
    select_device,
    set_global_seed,
)
from src.config import ROOT_DIR, SIMOPA_PATH, STUDENT_MID_PATH
from src.student_opa_mid import SimOPAStudentMid


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone()
            for k, v in (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict()
        for k, v in state.items():
            src = v.detach()
            dst = self.shadow[k]
            # EMA interpolation only applies to floating-point tensors.
            # Integer/bool buffers (e.g., num_batches_tracked) should be copied directly.
            if torch.is_floating_point(dst):
                dst.mul_(self.decay).add_(src.to(dtype=dst.dtype), alpha=1.0 - self.decay)
            else:
                dst.copy_(src)

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}


def set_trainable_by_stage(model: SimOPAStudentMid, stage: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if stage == "head_only":
        for m in [model.layer4, model.feature_proj, model.prediction_head]:
            for p in m.parameters():
                p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True


def run_epoch(
    model: SimOPAStudentMid,
    teacher: SimOPATeacherWithFeatures,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    temperature: float,
    alpha_kd: float,
    beta_feat: float,
    gamma_rank: float,
    distill_warmup_epochs: int,
    epoch_idx: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_channels_last: bool = False,
    optimizer: torch.optim.Optimizer | None = None,
    ema: ModelEMA | None = None,
) -> tuple[float, float, dict]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total = 0
    ce_sum = 0.0
    kd_sum = 0.0
    feat_sum = 0.0
    rank_sum = 0.0
    rank_active = 0
    batch_count = 0
    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        for x, y, group_keys in dataloader:
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
                    t_logits = t_logits.detach().clone()
                    t_feat = t_feat.detach().clone()
                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                s_logits, s_feat = model(x, return_features=True)
                loss_ce = criterion(s_logits, y)
                loss_kd = kd_loss(s_logits, t_logits, temperature=temperature)
                loss_feat = feature_distill_loss(s_feat, t_feat)
                loss_rank = rank_distill_loss(
                    torch.softmax(s_logits, dim=-1)[:, 1],
                    torch.softmax(t_logits, dim=-1)[:, 1],
                    list(group_keys),
                )
                if float(loss_rank.detach().item()) > 1e-8:
                    rank_active += 1
                if epoch_idx <= max(0, int(distill_warmup_epochs)):
                    beta_eff, gamma_eff = 0.0, 0.0
                else:
                    beta_eff, gamma_eff = beta_feat, gamma_rank
                loss = loss_ce + alpha_kd * loss_kd + beta_eff * loss_feat + gamma_eff * loss_rank
            if is_train:
                if scaler is not None and use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if ema is not None:
                    ema.update(model)
            b = y.size(0)
            batch_count += 1
            total += b
            total_loss += float(loss.item()) * b
            total_correct += int((s_logits.argmax(dim=1) == y).sum().item())
            ce_sum += float(loss_ce.detach().item()) * b
            kd_sum += float(loss_kd.detach().item()) * b
            feat_sum += float(loss_feat.detach().item()) * b
            rank_sum += float(loss_rank.detach().item()) * b
    return (
        total_loss / max(1, total),
        total_correct / max(1, total),
        {
            "loss_ce": ce_sum / max(1, total),
            "loss_kd": kd_sum / max(1, total),
            "loss_feat": feat_sum / max(1, total),
            "loss_rank": rank_sum / max(1, total),
            "rank_active_ratio": float(rank_active) / max(1, batch_count),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 5-8M mid-size OPA student.")
    parser.add_argument("--data-root", type=Path, default=ROOT_DIR / "new_OPA")
    parser.add_argument("--train-csv", type=Path, default=ROOT_DIR / "new_OPA" / "train_set.csv")
    parser.add_argument("--val-csv", type=Path, default=ROOT_DIR / "new_OPA" / "val_set.csv")
    parser.add_argument("--simopa-weight", type=Path, default=SIMOPA_PATH)
    parser.add_argument("--output", type=Path, default=STUDENT_MID_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=2.5)
    parser.add_argument("--alpha-kd", type=float, default=0.5)
    parser.add_argument("--beta-feat", type=float, default=0.10)
    parser.add_argument("--gamma-rank", type=float, default=0.03)
    parser.add_argument("--distill-warmup-epochs", type=int, default=3)
    parser.add_argument("--freeze-head-epochs", type=int, default=2)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=ROOT_DIR / "logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_name:
        args.run_name = time.strftime("student_mid_%Y%m%d_%H%M%S")
    set_global_seed(int(args.seed), deterministic=bool(args.deterministic))
    train_csv, val_csv = prepare_train_val_csv(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    device = select_device(args.device)
    use_amp = (device.type == "cuda") and (not args.disable_amp)
    amp_dtype = torch.float16
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not bool(args.deterministic)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
        loader_kwargs["persistent_workers"] = not args.no_persistent_workers

    train_dataset = OPACsvDataset(args.data_root, train_csv, image_size=args.image_size, max_samples=args.max_train_samples)
    val_dataset = OPACsvDataset(args.data_root, val_csv, image_size=args.image_size, max_samples=args.max_val_samples)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    teacher = SimOPATeacherWithFeatures(args.simopa_weight).to(device)
    model = SimOPAStudentMid().to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        teacher = teacher.to(memory_format=torch.channels_last)
    if args.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"torch.compile enabled for student mid: mode={args.compile_mode}")
        except Exception as exc:
            print(f"torch.compile unavailable, fallback eager: {exc}")

    class_w = train_dataset.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    set_trainable_by_stage(model._orig_mod if hasattr(model, "_orig_mod") else model, "head_only")
    optimizer = torch.optim.AdamW(
        (p for p in (model._orig_mod if hasattr(model, "_orig_mod") else model).parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.1)
    ema = ModelEMA(model, decay=args.ema_decay)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_dir, metrics_csv = _setup_run_logging(
        log_dir=args.log_dir,
        run_name=args.run_name,
        config={
            "script": "train_student_mid.py",
            "device": str(device),
            "output": str(args.output),
            "args": vars(args),
        },
    )
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    else:
        scaler = None

    best_val_acc = -1.0
    best_state = None
    print(f"device={device} log_dir={run_dir}")
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        if epoch == args.freeze_head_epochs + 1:
            set_trainable_by_stage(base_model, "full")
            current_lr = float(optimizer.param_groups[0]["lr"])
            unfreeze_lr = max(args.lr * 0.2, current_lr * 0.5)
            optimizer = torch.optim.AdamW(base_model.parameters(), lr=unfreeze_lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - epoch + 1), eta_min=args.lr * 0.1
            )

        train_loss, train_acc, train_aux = run_epoch(
            model=model,
            teacher=teacher,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            temperature=args.temperature,
            alpha_kd=args.alpha_kd,
            beta_feat=args.beta_feat,
            gamma_rank=args.gamma_rank,
            distill_warmup_epochs=args.distill_warmup_epochs,
            epoch_idx=epoch,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            use_channels_last=bool(args.channels_last),
            optimizer=optimizer,
            ema=ema,
        )
        val_loss, val_acc, val_aux = run_epoch(
            model=model,
            teacher=teacher,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            temperature=args.temperature,
            alpha_kd=args.alpha_kd,
            beta_feat=args.beta_feat,
            gamma_rank=args.gamma_rank,
            distill_warmup_epochs=args.distill_warmup_epochs,
            epoch_idx=epoch,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=None,
            use_channels_last=bool(args.channels_last),
        )
        scheduler.step()

        # quick EMA eval (classification only)
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        cur_state = base_model.state_dict()
        base_model.load_state_dict(ema.state_dict(), strict=True)
        _, ema_val_acc, _ = run_epoch(
            model=model,
            teacher=teacher,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            temperature=args.temperature,
            alpha_kd=args.alpha_kd,
            beta_feat=args.beta_feat,
            gamma_rank=args.gamma_rank,
            distill_warmup_epochs=args.distill_warmup_epochs,
            epoch_idx=epoch,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=None,
            use_channels_last=bool(args.channels_last),
        )
        base_model.load_state_dict(cur_state, strict=True)
        effective_val = max(val_acc, ema_val_acc)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} ema_val_acc={ema_val_acc:.4f} "
            f"rank_active={train_aux['rank_active_ratio']:.3f}"
        )
        if effective_val > best_val_acc:
            best_val_acc = effective_val
            best_state = ema.state_dict() if ema_val_acc >= val_acc else cur_state
            torch.save(
                {
                    "model": "SimOPAStudentMid",
                    "model_state_dict": best_state,
                    "image_size": args.image_size,
                    "val_acc": effective_val,
                    "epoch": epoch,
                    "student_arch": "ResNet18-4ch-width0.75",
                    "temperature": float(args.temperature),
                    "alpha_kd": float(args.alpha_kd),
                    "beta_feat": float(args.beta_feat),
                    "gamma_rank": float(args.gamma_rank),
                },
                args.output,
            )
            print(f"saved_best={args.output}")

        epoch_time = time.time() - t_epoch
        _append_metrics_row(
            metrics_csv,
            {
                "epoch": int(epoch),
                "epoch_time_sec": f"{epoch_time:.3f}",
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{effective_val:.6f}",
                "best_val_acc": f"{best_val_acc:.6f}",
                "lr": f"{float(optimizer.param_groups[0]['lr']):.8f}",
                "train_loss_ce": f"{train_aux['loss_ce']:.6f}",
                "train_loss_kd": f"{train_aux['loss_kd']:.6f}",
                "train_loss_feat": f"{train_aux['loss_feat']:.6f}",
                "train_loss_rank": f"{train_aux['loss_rank']:.6f}",
                "train_rank_active_ratio": f"{train_aux['rank_active_ratio']:.6f}",
                "val_loss_ce": f"{val_aux['loss_ce']:.6f}",
                "val_loss_kd": f"{val_aux['loss_kd']:.6f}",
                "val_loss_feat": f"{val_aux['loss_feat']:.6f}",
                "val_loss_rank": f"{val_aux['loss_rank']:.6f}",
                "val_rank_active_ratio": f"{val_aux['rank_active_ratio']:.6f}",
            },
        )

    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_val_acc": float(best_val_acc),
                "output_weight": str(args.output),
                "metrics_csv": str(metrics_csv),
                "train_csv_used": str(train_csv),
                "val_csv_used": str(val_csv),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
