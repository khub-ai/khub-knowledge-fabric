"""
elicit_from_failures.py

Show Opus the remaining failure images, ask for subtype + corrective rule.
Compact prompt — minimal tokens.

Usage:
  python elicit_from_failures.py
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

# ── config ──────────────────────────────────────────────────────────────────
MODEL       = "claude-opus-4-5"
DATA_DIR    = r"C:\_backup\ml\data\DermaMNIST_HAM10000"
FAILURE_IDS = [
    "ISIC_0024410",
    "ISIC_0024647",
    "ISIC_0024911",
    "ISIC_0025128",
]
# ────────────────────────────────────────────────────────────────────────────

PROMPT = """\
Context: a cheaper AI model saw this dermoscopic image and called it \
Melanocytic Nevus. It is actually Melanoma. The model missed it.

Reply in JSON only:
{
  "subtype": "<superficial_spreading|nodular|lentigo_maligna|amelanotic|acral|other>",
  "subtype_confidence": "<high|medium|low>",
  "why_model_failed": "<one sentence>",
  "key_feature": "<single most discriminating visible feature>",
  "rule_condition": "<complete rule preconditions, semicolon-separated>",
  "rule_action": "classify as Melanoma"
}

Keep rule_condition concise (2-4 preconditions max). Only include features \
that are clearly visible in this image."""


def _b64(path):
    return base64.standard_b64encode(Path(path).read_bytes()).decode()


def main():
    # keys
    kf = Path("P:/_access/Security/api_keys.env")
    for line in kf.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip(); v = v.strip()
            if k == "ANTHROPIC_API_KEY" and not os.environ.get(k):
                os.environ[k] = v

    import anthropic
    client = anthropic.Anthropic()

    ds = load_ham10000(DATA_DIR)
    # build image_id -> file_path map
    img_map = {img.image_id: str(img.file_path)
               for img in ds.sample_images("mel", 200, split="test", seed=0)
               + ds.sample_images("mel", 200, split="train", seed=0)}

    rules = []
    for iid in FAILURE_IDS:
        path = img_map.get(iid)
        if not path:
            console.print(f"[red]Not found:[/red] {iid}")
            continue

        console.print(f"\n[bold]{iid}[/bold]")
        resp = client.messages.create(
            model=MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": _b64(path)}},
                {"type": "text", "text": PROMPT},
            ]}]
        )
        raw = resp.content[0].text
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(m.group()) if m else {"raw": raw}
        parsed["image_id"] = iid

        console.print(f"  subtype: [yellow]{parsed.get('subtype','?')}[/yellow] "
                      f"({parsed.get('subtype_confidence','?')})")
        console.print(f"  why failed: {parsed.get('why_model_failed','')}")
        console.print(f"  key feature: [cyan]{parsed.get('key_feature','')}")
        console.print(f"  rule: {parsed.get('rule_condition','')[:120]}")
        console.print(f"  tokens: in={resp.usage.input_tokens} "
                      f"out={resp.usage.output_tokens}")
        rules.append(parsed)

    out = _HERE / "elicited_rules_from_failures.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)
    console.print(f"\nSaved to [cyan]{out.name}[/cyan]")


if __name__ == "__main__":
    main()
