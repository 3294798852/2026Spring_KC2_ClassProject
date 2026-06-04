import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compress import compress_student_model
from src.train import train_teacher_student


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain and compress placement model.")
    parser.add_argument(
        "--profile",
        choices=["quick", "standard"],
        default="standard",
        help="quick for debugging, standard for better quality",
    )
    args = parser.parse_args()

    train_kwargs = (
        dict(epochs_teacher=2, epochs_student=3, train_size=1800, val_size=300, batch_size=24)
        if args.profile == "quick"
        else dict(epochs_teacher=4, epochs_student=6, train_size=4200, val_size=800, batch_size=32)
    )

    print("[1/2] training teacher+student...")
    res = train_teacher_student(device="cpu", **train_kwargs)
    print(
        f"teacher_loss={res.teacher_loss:.4f}, student_loss={res.student_loss:.4f}, "
        f"val_mae={res.student_val_mae:.4f}, val_corr={res.student_val_corr:.4f}"
    )
    print("[2/2] compressing student...")
    src_mb, dst_mb = compress_student_model()
    print(f"model size: {src_mb:.2f}MB -> {dst_mb:.2f}MB")
    print("done.")
