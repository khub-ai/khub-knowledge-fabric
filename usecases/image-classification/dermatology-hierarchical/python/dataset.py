"""
dataset.py — Hierarchical group definitions for the 2-level derm pipeline.

Level 1: 5 coarse groups — Melanocytic, Keratosis-type, Basal Cell Carcinoma,
         Vascular Lesion, Dermatofibroma.
Level 2: fine classification within multi-class groups only.
         Melanocytic -> Melanoma vs Melanocytic Nevus
         Keratosis   -> Benign Keratosis vs Actinic Keratosis
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fine-grained class list (same as multiclass) — used by dataset loader
# ---------------------------------------------------------------------------

ALL_CLASSES = [
    {"dx": "mel",   "name": "Melanoma"},
    {"dx": "nv",    "name": "Melanocytic Nevus"},
    {"dx": "bkl",   "name": "Benign Keratosis"},
    {"dx": "bcc",   "name": "Basal Cell Carcinoma"},
    {"dx": "akiec", "name": "Actinic Keratosis"},
    {"dx": "vasc",  "name": "Vascular Lesion"},
    {"dx": "df",    "name": "Dermatofibroma"},
]
CATEGORY_NAMES  = [c["name"] for c in ALL_CLASSES]
DX_CODES        = [c["dx"]   for c in ALL_CLASSES]
DX_TO_NAME      = {c["dx"]: c["name"] for c in ALL_CLASSES}
NAME_TO_DX      = {c["name"]: c["dx"] for c in ALL_CLASSES}
CATEGORY_SET_ID = "dermatology_7class"   # compat alias for downstream tools

# ---------------------------------------------------------------------------
# Level-1 groups
# ---------------------------------------------------------------------------

LEVEL1_GROUPS = [
    {
        "id":      "melanocytic",
        "name":    "Melanocytic",
        "classes": ["Melanoma", "Melanocytic Nevus"],
    },
    {
        "id":      "keratosis",
        "name":    "Keratosis-type",
        "classes": ["Benign Keratosis", "Actinic Keratosis"],
    },
    {
        "id":      "bcc",
        "name":    "Basal Cell Carcinoma",
        "classes": ["Basal Cell Carcinoma"],
    },
    {
        "id":      "vascular",
        "name":    "Vascular Lesion",
        "classes": ["Vascular Lesion"],
    },
    {
        "id":      "dermatofibroma",
        "name":    "Dermatofibroma",
        "classes": ["Dermatofibroma"],
    },
]

LEVEL1_CATEGORY_SET_ID = "derm_h_level1"
LEVEL1_GROUP_NAMES     = [g["name"] for g in LEVEL1_GROUPS]
LEVEL1_GROUP_IDS       = [g["id"]   for g in LEVEL1_GROUPS]

# ---------------------------------------------------------------------------
# Level-2 sub-problems (only multi-class groups)
# ---------------------------------------------------------------------------

LEVEL2_SUBPROBLEMS: dict[str, dict] = {
    "melanocytic": {
        "category_set_id": "derm_h_l2_melanocytic",
        "categories":      ["Melanoma", "Melanocytic Nevus"],
    },
    "keratosis": {
        "category_set_id": "derm_h_l2_keratosis",
        "categories":      ["Benign Keratosis", "Actinic Keratosis"],
    },
}

# ---------------------------------------------------------------------------
# Lookup maps
# ---------------------------------------------------------------------------

CLASS_TO_GROUP_ID:   dict[str, str] = {}
CLASS_TO_GROUP_NAME: dict[str, str] = {}
GROUP_ID_TO_NAME:    dict[str, str] = {}
GROUP_NAME_TO_ID:    dict[str, str] = {}
GROUP_TO_SOLO_CLASS: dict[str, str] = {}   # group_id -> class name, solo groups only

for _g in LEVEL1_GROUPS:
    GROUP_ID_TO_NAME[_g["id"]]   = _g["name"]
    GROUP_NAME_TO_ID[_g["name"]] = _g["id"]
    for _cls in _g["classes"]:
        CLASS_TO_GROUP_ID[_cls]   = _g["id"]
        CLASS_TO_GROUP_NAME[_cls] = _g["name"]
    if len(_g["classes"]) == 1:
        GROUP_TO_SOLO_CLASS[_g["id"]] = _g["classes"][0]
