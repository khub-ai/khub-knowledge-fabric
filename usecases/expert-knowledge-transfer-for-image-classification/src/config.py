"""
Experiment configuration for UC-02: Expert Knowledge Transfer for Image Classification
"""

import os
from pathlib import Path

# --- Dataset paths ---
DATA_ROOT = Path(r"C:\_backup\ml\data\_tmp\CUB_200_2011\CUB_200_2011")
IMAGES_DIR        = DATA_ROOT / "images"
CLASSES_FILE      = DATA_ROOT / "classes.txt"
IMAGES_FILE       = DATA_ROOT / "images.txt"
LABELS_FILE       = DATA_ROOT / "image_class_labels.txt"
SPLIT_FILE        = DATA_ROOT / "train_test_split.txt"
BBOXES_FILE       = DATA_ROOT / "bounding_boxes.txt"
ATTRIBUTES_FILE   = DATA_ROOT / "attributes" / "class_attribute_labels_continuous.txt"
IMG_ATTRIBUTES    = DATA_ROOT / "attributes" / "image_attribute_labels.txt"
PARTS_FILE        = DATA_ROOT / "parts" / "parts.txt"
PART_LOCS_FILE    = DATA_ROOT / "parts" / "part_locs.txt"

# --- Project paths ---
PROJECT_ROOT       = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
TEACHING_DIR       = PROJECT_ROOT / "teaching_sessions"
RESULTS_DIR        = PROJECT_ROOT / "results"

# --- API ---
# Keys stored in P:/_access/Security/api_keys.env as KEY=VALUE pairs
API_KEYS_ENV_PATH = Path("P:/_access/Security/api_keys.env")
OPENAI_MODEL      = "gpt-4-vision-preview"
OPENAI_MAX_TOKENS = 1024

# --- Experiment settings ---
# Number of confusable pairs to include in the primary test set
N_CONFUSABLE_PAIRS = 15

# Max images per class per condition (None = use all)
MAX_TEST_IMAGES_PER_CLASS = 20

# Random seed for reproducibility
SEED = 42


def _load_env_file(path: Path) -> dict:
    """Parse a KEY=VALUE env file, ignoring comments and blank lines."""
    result = {}
    if not path.exists():
        return result
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                result[k.strip()] = v.strip()
    return result


def get_openai_key() -> str:
    # 1. Environment variable takes priority
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    # 2. P drive env file
    env = _load_env_file(API_KEYS_ENV_PATH)
    key = env.get("OPENAI_API_KEY")
    if key:
        return key
    raise FileNotFoundError(
        f"OpenAI API key not found. Set OPENAI_API_KEY env var "
        f"or ensure {API_KEYS_ENV_PATH} contains OPENAI_API_KEY=..."
    )
