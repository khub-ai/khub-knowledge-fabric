"""
probe_tutor_subtype.py

Show Claude Opus a small set of melanoma images and ask it to identify
the dermoscopic subtype and key distinguishing features. This is a
prerequisite check: does the tutor model have the subtype knowledge
we need before investing in proactive rule authoring?

Usage:
  python probe_tutor_subtype.py --n-images 5 --model claude-opus-4-5
"""
from __future__ import annotations
import argparse, base64, json, os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[4]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

from dataset import load as load_ham10000
from rich.console import Console
from rich.panel import Panel

console = Console()


def _load_api_keys():
    key_file = Path("P:/_access/Security/api_keys.env")
    if key_file.exists():
        for line in key_file.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip(); v = v.strip()
                if k in ("ANTHROPIC_API_KEY",) and not os.environ.get(k):
                    os.environ[k] = v


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


PROMPT = """You are a senior dermoscopist. Look at this dermoscopic image carefully.

Please answer the following questions in JSON format:

{
  "melanoma_subtype": "<superficial_spreading | nodular | lentigo_maligna | amelanotic | acral_lentiginous | unknown>",
  "confidence_in_subtype": "<high | medium | low>",
  "subtype_rationale": "<1-2 sentences explaining which features led you to this subtype>",
  "key_features_visible": ["<feature 1>", "<feature 2>", ...],
  "features_absent": ["<feature 1>", ...],
  "classic_ABCD_summary": "<brief summary of asymmetry/border/color/diameter observations>",
  "overall_diagnosis": "<Melanoma | Melanocytic Nevus | Uncertain>",
  "diagnosis_confidence": "<high | medium | low>"
}

Be precise. If the subtype is not clearly identifiable, say so with low confidence."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--model", default="claude-opus-4-5")
    parser.add_argument("--data-dir",
                        default=r"C:\_backup\ml\data\DermaMNIST_HAM10000")
    parser.add_argument("--output", default="probe_tutor_subtype_results.json")
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    _load_api_keys()
    import anthropic
    client = anthropic.Anthropic()

    ds = load_ham10000(args.data_dir)
    # Sample both melanoma and nevus to see if tutor distinguishes
    mel_imgs = ds.sample_images("mel", args.n_images, split="test", seed=args.seed)
    nv_imgs  = ds.sample_images("nv",  args.n_images, split="test", seed=args.seed)
    all_imgs = [(img, "Melanoma") for img in mel_imgs] + \
               [(img, "Melanocytic Nevus") for img in nv_imgs]

    console.print(f"Model: [cyan]{args.model}[/cyan]")
    console.print(f"Probing {len(all_imgs)} images "
                  f"({args.n_images} melanoma + {args.n_images} nevus)\n")

    results = []
    for img_obj, true_label in all_imgs:
        img_path = str(img_obj.file_path)
        image_id = img_obj.image_id
        console.print(f"[dim]{image_id}[/dim]  gt=[bold]{true_label}[/bold]")

        img_b64 = _encode_image(img_path)
        response = client.messages.create(
            model=args.model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }]
        )

        raw = response.content[0].text
        # Parse JSON
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group()) if match else {"raw": raw}
        parsed["image_id"] = image_id
        parsed["true_label"] = true_label
        parsed["correct_diagnosis"] = (
            parsed.get("overall_diagnosis", "") == true_label
        )

        subtype = parsed.get("melanoma_subtype", "—")
        conf = parsed.get("confidence_in_subtype", "—")
        diag = parsed.get("overall_diagnosis", "—")
        correct = parsed["correct_diagnosis"]
        mark = "[green]correct[/green]" if correct else "[red]WRONG[/red]"

        console.print(
            f"  subtype=[yellow]{subtype}[/yellow] ({conf})  "
            f"diagnosis={diag}  {mark}"
        )
        console.print(
            f"  rationale: {parsed.get('subtype_rationale','')[:100]}"
        )
        console.print()
        results.append(parsed)

    # Summary
    mel_results = [r for r in results if r["true_label"] == "Melanoma"]
    subtypes = [r.get("melanoma_subtype","unknown") for r in mel_results]
    from collections import Counter
    console.print("\n[bold]Melanoma subtype distribution (as seen by tutor):[/bold]")
    for st, count in Counter(subtypes).most_common():
        console.print(f"  {st}: {count}")

    correct_count = sum(1 for r in results if r["correct_diagnosis"])
    console.print(f"\nDiagnosis accuracy: {correct_count}/{len(results)}")

    high_conf_subtype = sum(
        1 for r in mel_results
        if r.get("confidence_in_subtype") == "high"
    )
    console.print(
        f"High-confidence subtype calls on melanoma: "
        f"{high_conf_subtype}/{len(mel_results)}"
    )

    # Save
    out_path = _HERE / args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    console.print(f"\nResults saved to [cyan]{args.output}[/cyan]")


if __name__ == "__main__":
    main()
