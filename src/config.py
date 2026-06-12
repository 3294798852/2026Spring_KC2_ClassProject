from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

SIMOPA_PATH = MODEL_DIR / "SimOPA.pth"
STUDENT_CNN_PATH = MODEL_DIR / "student_cnn.pth"
STUDENT_DUAL_PATH = MODEL_DIR / "student_dual_geom.pth"
STUDENT_MID_PATH = MODEL_DIR / "student_mid.pth"
