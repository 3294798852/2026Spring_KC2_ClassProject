from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

TEACHER_PATH = MODEL_DIR / "teacher.pth"
STUDENT_PATH = MODEL_DIR / "student.pth"
COMPRESSED_PATH = MODEL_DIR / "student_compressed.pth"
SIMOPA_PATH = MODEL_DIR / "SimOPA.pth"
