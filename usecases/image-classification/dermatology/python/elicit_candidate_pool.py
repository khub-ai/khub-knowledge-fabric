"""
elicit_candidate_pool.py

For each failure image, ask Opus to generate 4 candidate rule hypotheses
at increasing specificity. More specific = fewer features but harder to satisfy.
The validation gate will filter; we just need one good one per failure.

Compact prompt — ~600 tokens in per image.
"""
import base64, json, os, re, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[4]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

from dataset import load as load_ham10000
from rich.console import Console
console = Console()

MODEL    = "claude-opus-4-5"
DATA_DIR = r"C:\_backup\ml\data\DermaMNIST_HAM10000"
FAILURE_IDS = ["ISIC_0024410", "ISIC_0024647", "ISIC_0024911", "ISIC_0025128"]

PROMPT = """\
This dermoscopic image is a Melanoma, but a cheaper AI called it Melanocytic Nevus.

Generate 4 candidate rules to fix this, from least specific to most specific.
Each rule must fire on THIS image but ideally NOT on benign moles.

Reply in JSON only — array of 4 objects:
[
  {
    "level": 1,
    "specificity": "broad",
    "condition": "<1 key feature only — the single strongest melanoma signal visible>",
    "confidence_fires_here": "<high|medium|low>"
  },
  {
    "level": 2,
    "specificity": "moderate",
    "condition": "<2 features that together are more specific to melanoma>",
    "confidence_fires_here": "<high|medium|low>"
  },
  {
    "level": 3,
    "specificity": "specific",
    "condition": "<3 features — more restrictive, fewer false positives expected>",
    "confidence_fires_here": "<high|medium|low>"
  },
  {
    "level": 4,
    "specificity": "most_specific",
    "condition": "<3-4 features — the combination that most precisely describes THIS lesion>",
    "confidence_fires_here": "<high|medium|low>"
  }
]

Conditions must describe only what is CLEARLY VISIBLE in this image.
Use precise dermoscopic terminology. Keep each condition concise (no sentences)."""


def _b64(path):
    return base64.standard_b64encode(Path(path).read_bytes()).decode()


def main():
    kf = Path("P:/_access/Security/api_keys.env")
    for line in kf.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            if k.strip() == "ANTHROPIC_API_KEY" and not os.environ.get(k.strip()):
                os.environ[k.strip()] = v.strip()

    import anthropic
    client = anthropic.Anthropic()

    ds = load_ham10000(DATA_DIR)
    all_mel = (ds.sample_images("mel", 500, split="test", seed=0) +
               ds.sample_images("mel", 500, split="train", seed=0))
    img_map = {img.image_id: str(img.file_path) for img in all_mel}

    all_candidates = []
    total_in = total_out = 0

    for iid in FAILURE_IDS:
        path = img_map.get(iid)
        if not path:
            console.print(f"[red]Not found: {iid}[/red]")
            continue

        console.print(f"\n[bold]{iid}[/bold]")
        resp = client.messages.create(
            model=MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": _b64(path)}},
                {"type": "text", "text": PROMPT},
            ]}]
        )
        raw = resp.content[0].text
        total_in  += resp.usage.input_tokens
        total_out += resp.usage.output_tokens

        m = re.search(r"\[.*\]", raw, re.DOTALL)
        levels = json.loads(m.group()) if m else []

        for lvl in levels:
            rule_id = f"r_{iid}_L{lvl['level']}"
            cond = lvl.get("condition", "")
            console.print(f"  L{lvl['level']} [{lvl['specificity']}]: {cond[:90]}")
            all_candidates.append({
                "rule_id": rule_id,
                "source_image": iid,
                "level": lvl["level"],
                "specificity": lvl["specificity"],
                "description": f"{iid} L{lvl['level']} ({lvl['specificity']})",
                "condition": (
                    f"[Patch rule \u2014 melanoma_vs_melanocytic_nevus] {cond}"
                ),
                "action": "classify as Melanoma",
                "confidence_fires_here": lvl.get("confidence_fires_here", ""),
                "source": "candidate_pool_claude_opus",
                "subtype": "melanoma",
            })

        console.print(f"  tokens: in={resp.usage.input_tokens} out={resp.usage.output_tokens}")

    console.print(f"\nTotal candidates: {len(all_candidates)}")
    console.print(f"Total tokens: in={total_in} out={total_out} "
                  f"(~${(total_in*5 + total_out*25)/1e6:.4f} at Opus rates)")

    out = _HERE / "patch_rules_candidate_pool.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, indent=2, ensure_ascii=False)
    console.print(f"Saved to [cyan]{out.name}[/cyan]")


if __name__ == "__main__":
    main()
