"""
elicit_nodular_rules.py

Proactive knowledge elicitation experiment.
Queries Claude Opus to author rules for nodular melanoma subtype
WITHOUT needing failure cases first. Rules are then validated
through the standard KF precision gate.
"""
import os, sys, json
from pathlib import Path

# Load API keys
key_file = Path("P:/_access/Security/api_keys.env")
for line in key_file.read_text().splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        k = k.strip(); v = v.strip()
        if k in ("ANTHROPIC_API_KEY",) and not os.environ.get(k):
            os.environ[k] = v

import anthropic
client = anthropic.Anthropic()

PROMPT = """You are a senior dermoscopist with deep expertise in melanoma subtypes.

I am building a rule-based system to help a weaker AI model correctly identify \
melanoma vs melanocytic nevus in dermoscopic images. The system works by injecting \
explicit rules with hard-gate preconditions. A rule fires ONLY when ALL its \
preconditions are confirmed present in the image.

The current rules cover superficial spreading melanoma (regression structures, \
peppering, peripheral globules). I need NEW rules specifically for NODULAR MELANOMA, \
which has a very different dermoscopic appearance and is frequently missed.

Please author 3 distinct rules for nodular melanoma. Each rule must:
1. Target a specific, reliable dermoscopic feature cluster of nodular melanoma
2. Have preconditions specific enough to NOT fire on benign melanocytic nevi
3. Be based on established dermoscopic criteria (EFG rule, blue-black homogeneous \
   pattern, milky-red areas, dotted vessels, etc.)

Return ONLY a JSON array of 3 rules in this exact format:
[
  {
    "rule_id": "r_nodular_001",
    "description": "one sentence description",
    "condition": "[Patch rule - melanoma_vs_melanocytic_nevus] 1. precondition one; 2. precondition two; 3. precondition three",
    "action": "classify as Melanoma",
    "rationale": "brief clinical rationale citing the dermoscopic principle",
    "subtype": "nodular_melanoma",
    "source": "proactive_elicitation_claude_opus"
  }
]

Be precise and clinically accurate. Use established dermoscopy terminology.
These rules will be validated against real ISIC dermoscopic images."""

print("Querying Claude Opus for nodular melanoma rules...")
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=2000,
    messages=[{"role": "user", "content": PROMPT}]
)

raw = response.content[0].text
print("\n--- Raw response ---")
print(raw)

# Extract JSON
import re
match = re.search(r"\[.*\]", raw, re.DOTALL)
if match:
    rules = json.loads(match.group())
    out_path = Path(__file__).parent / "patch_rules_proactive_nodular_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(rules)} rules to {out_path.name}")
else:
    print("\nCould not parse JSON from response")
