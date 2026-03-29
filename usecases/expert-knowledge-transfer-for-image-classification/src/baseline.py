"""
Baseline evaluation: zero-shot and few-shot GPT-4V classification.

For each confusable pair (A, B), present a test image to GPT-4V and ask it
to classify the bird as either species A or species B.

Conditions:
  - zero_shot : no additional context beyond the species names
  - few_shot  : 3 example images per class provided before the test image
  - kf_taught : rules from the KF knowledge base injected into the system prompt
                (this condition is driven by evaluator.py, not this module)
"""

from __future__ import annotations
import base64
import json
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent))

from config import get_openai_key, OPENAI_MODEL, OPENAI_MAX_TOKENS
from dataset import CUBDataset, CUBImage
from confusable_pairs import ConfusablePair

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai package required: pip install openai")


Condition = Literal["zero_shot", "few_shot"]


# ---------------------------------------------------------------------------

def _encode_image(path: Path) -> str:
    """Return base64-encoded JPEG string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_content(path: Path) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(path)}"},
    }


# ---------------------------------------------------------------------------

def classify_zero_shot(
    client: OpenAI,
    pair: ConfusablePair,
    test_image: CUBImage,
    rules_text: str | None = None,
) -> dict:
    """
    Ask GPT-4V to classify test_image as either species A or B.
    If rules_text is provided, inject it as expert context (kf_taught condition).
    """
    name_a = pair.class_name_a
    name_b = pair.class_name_b

    if rules_text:
        system_msg = (
            f"You are an expert ornithologist. "
            f"You will be shown a bird photo and must decide whether it is a "
            f"{name_a} or a {name_b}.\n\n"
            f"Use the following expert identification guidelines:\n\n"
            f"{rules_text}\n\n"
            f"Respond with ONLY a JSON object: "
            f'{{\"prediction\": \"<species name>\", \"confidence\": <0-1>, \"reasoning\": \"<one sentence>\"}}'
        )
    else:
        system_msg = (
            f"You are an expert ornithologist. "
            f"You will be shown a bird photo and must decide whether it is a "
            f"{name_a} or a {name_b}. "
            f"Respond with ONLY a JSON object: "
            f'{{\"prediction\": \"<species name>\", \"confidence\": <0-1>, \"reasoning\": \"<one sentence>\"}}'
        )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=OPENAI_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify this bird:"},
                    _image_content(test_image.file_path),
                ],
            },
        ],
    )
    raw = response.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"prediction": raw, "confidence": None, "reasoning": None}

    result["image_id"]   = test_image.image_id
    result["true_class"] = test_image.class_name
    result["condition"]  = "kf_taught" if rules_text else "zero_shot"
    return result


def classify_few_shot(
    client: OpenAI,
    pair: ConfusablePair,
    test_image: CUBImage,
    examples_a: list[CUBImage],
    examples_b: list[CUBImage],
) -> dict:
    """
    Provide example images for each class before classifying test_image.
    examples_a / examples_b: typically 3 training images per class.
    """
    name_a = pair.class_name_a
    name_b = pair.class_name_b

    system_msg = (
        f"You are an expert ornithologist. "
        f"You will first see example photos labelled as {name_a} or {name_b}, "
        f"then a test photo you must classify. "
        f"Respond with ONLY a JSON object: "
        f'{{\"prediction\": \"<species name>\", \"confidence\": <0-1>, \"reasoning\": \"<one sentence>\"}}'
    )

    # Build example content blocks
    example_blocks: list[dict] = []
    for img in examples_a:
        example_blocks.append({"type": "text", "text": f"Example — {name_a}:"})
        example_blocks.append(_image_content(img.file_path))
    for img in examples_b:
        example_blocks.append({"type": "text", "text": f"Example — {name_b}:"})
        example_blocks.append(_image_content(img.file_path))

    example_blocks.append({"type": "text", "text": "Now classify this bird:"})
    example_blocks.append(_image_content(test_image.file_path))

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=OPENAI_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": example_blocks},
        ],
    )
    raw = response.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"prediction": raw, "confidence": None, "reasoning": None}

    result["image_id"]   = test_image.image_id
    result["true_class"] = test_image.class_name
    result["condition"]  = "few_shot"
    return result


# ---------------------------------------------------------------------------

def build_client() -> OpenAI:
    return OpenAI(api_key=get_openai_key())
