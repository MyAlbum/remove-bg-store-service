import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_ROOT = BASE_DIR / "app" / "static"
EXTERNAL_CUTOUT_DIR = STATIC_ROOT / "generated-assets" / "external-cutouts"

REMOVE_BG_LOCAL_URL = os.getenv("REMOVE_BG_LOCAL_URL", "http://127.0.0.1:7860").rstrip("/")
NEXT_PUBLIC_URL = os.getenv("NEXT_PUBLIC_URL", "").rstrip("/")
REMOVE_BG_PUBLIC_BASE_URL = os.getenv("REMOVE_BG_PUBLIC_BASE_URL", "").rstrip("/")

DEFAULT_EXTERNAL_CUTOUT_MODEL = os.getenv("REMOVE_BG_MODEL", "bria-rmbg-2.0")
DEFAULT_REMOVE_BG_ALPHA_MATTING_FG_THRESHOLD = int(
    os.getenv("NEXT_PUBLIC_REMOVE_BG_ALPHA_MATTING_FG_THRESHOLD", "200")
)
DEFAULT_REMOVE_BG_ALPHA_MATTING_BG_THRESHOLD = int(
    os.getenv("NEXT_PUBLIC_REMOVE_BG_ALPHA_MATTING_BG_THRESHOLD", "40")
)
DEFAULT_REMOVE_BG_ALPHA_MATTING_ERODE_SIZE = int(
    os.getenv("NEXT_PUBLIC_REMOVE_BG_ALPHA_MATTING_ERODE_SIZE", "8")
)
DEFAULT_REMOVE_BG_EDGE_FEATHER_PX = float(
    os.getenv("NEXT_PUBLIC_REMOVE_BG_EDGE_FEATHER_PX", "0.2")
)


def ensure_external_cutout_dir() -> Path:
    EXTERNAL_CUTOUT_DIR.mkdir(parents=True, exist_ok=True)
    return EXTERNAL_CUTOUT_DIR
