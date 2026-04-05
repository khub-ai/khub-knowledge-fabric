"""
KF Teaching Session for UC-02.

Implements the SOLVER → SUPERVISOR → HUMAN escalation loop for rule
extraction and verification. The expert (HUMAN) is prompted interactively
when KF is uncertain about an extracted rule.

Flow:
  1. Load field-guide teaching text for a species pair
  2. SOLVER extracts discrete, verifiable rules via GPT-4
  3. Each rule is presented to the expert for confirmation / correction
  4. Confirmed rules are stored in the knowledge base as JSON
  5. On classification, rules are retrieved and injected as context
"""

from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from config import get_openai_key, KNOWLEDGE_BASE_DIR, TEACHING_DIR
from confusable_pairs import ConfusablePair

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai package required: pip install openai")


# ---------------------------------------------------------------------------
# Rule schema
# ---------------------------------------------------------------------------

def _make_rule(
    pair: ConfusablePair,
    rule_text: str,
    feature: str,
    favors: str,
    confidence: str,
    source: str,
    verified_by: str,
) -> dict[str, Any]:
    return {
        "pair": f"{pair.class_name_a} vs {pair.class_name_b}",
        "class_id_a": pair.class_id_a,
        "class_id_b": pair.class_id_b,
        "rule": rule_text,
        "feature": feature,
        "favors": favors,                # which species this rule points toward
        "confidence": confidence,        # "high" | "medium" | "low"
        "source": source,                # e.g. "Sibley Guide p.312"
        "verified_by": verified_by,      # "expert" | "auto"
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# SOLVER: extract rules from teaching text
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """
You are a Knowledge Fabric rule extractor. Given a block of expert ornithological
text describing how to distinguish species A from species B, extract a list of
discrete, independently applicable identification rules.

IMPORTANT: This is a VISUAL classification task. Extract ALL visual features
mentioned — size, shape, bill, eye, leg color, tail pattern, plumage, wingbars,
rump, facial markings, etc. — even if they are described as secondary or subtle.
Do NOT discard a rule just because it is not diagnostic on its own; supporting
and weak visual rules are still valuable for image classification.

For each rule output a JSON object with these fields:
  - rule        : one-sentence statement of the rule (e.g. "If the bill is as long as the head depth, it is a Hairy Woodpecker")
  - feature     : the visual feature referenced (e.g. "bill length", "eye color")
  - favors      : which species the rule points toward (use the exact species name given)
  - confidence  : "high" if the rule is diagnostic, "medium" if supporting, "low" if weak or subtle

Return a JSON array of rule objects. Nothing else.
""".strip()


def extract_rules(
    client: OpenAI,
    pair: ConfusablePair,
    teaching_text: str,
) -> list[dict]:
    """SOLVER step: extract candidate rules from field-guide text."""
    prompt = (
        f"Species A: {pair.class_name_a}\n"
        f"Species B: {pair.class_name_b}\n\n"
        f"Expert text:\n{teaching_text}"
    )
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        max_tokens=2048,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)

    # Unwrap to a list regardless of wrapper structure
    if isinstance(data, list):
        candidates = data
    else:
        # Try common wrapper keys first, then fall back to first value
        candidates = (
            data.get("rules")
            or data.get("items")
            or data.get("identification_rules")
            or next(iter(data.values()), [])
        )
        if not isinstance(candidates, list):
            candidates = [candidates]

    # Normalise: if GPT returned plain strings instead of dicts, wrap them
    normalised = []
    for item in candidates:
        if isinstance(item, str):
            normalised.append({"rule": item, "feature": "", "favors": "", "confidence": "medium"})
        elif isinstance(item, dict):
            normalised.append(item)
    return normalised


# ---------------------------------------------------------------------------
# HUMAN verification loop
# ---------------------------------------------------------------------------

def verify_rules_interactive(
    pair: ConfusablePair,
    candidate_rules: list[dict],
    teaching_text: str,
    source: str,
) -> list[dict]:
    """
    SUPERVISOR → HUMAN: present each extracted rule to the expert for
    confirmation, correction, or rejection.

    Returns the list of verified rules (with verified_by set).
    """
    print(f"\n{'='*70}")
    print(f"KF TEACHING SESSION: {pair.class_name_a} vs {pair.class_name_b}")
    print(f"{'='*70}")
    print(f"\nSource text:\n{teaching_text[:400]}{'...' if len(teaching_text) > 400 else ''}")
    print(f"\n{len(candidate_rules)} rules extracted. Please verify each.\n")

    verified = []
    for i, rule in enumerate(candidate_rules, 1):
        print(f"\n--- Rule {i}/{len(candidate_rules)} ---")
        print(f"  Rule    : {rule.get('rule', '')}")
        print(f"  Feature : {rule.get('feature', '')}")
        print(f"  Favors  : {rule.get('favors', '')}")
        print(f"  Conf.   : {rule.get('confidence', '')}")

        while True:
            choice = input("\n  [A]ccept / [E]dit / [R]eject / [?]explain  > ").strip().upper()
            if choice == "A":
                verified.append(_make_rule(
                    pair=pair,
                    rule_text=rule["rule"],
                    feature=rule.get("feature", ""),
                    favors=rule.get("favors", ""),
                    confidence=rule.get("confidence", "medium"),
                    source=source,
                    verified_by="expert",
                ))
                print("  [OK] Accepted.")
                break
            elif choice == "E":
                new_rule = input("  Enter corrected rule text: ").strip()
                verified.append(_make_rule(
                    pair=pair,
                    rule_text=new_rule,
                    feature=rule.get("feature", ""),
                    favors=rule.get("favors", ""),
                    confidence=rule.get("confidence", "medium"),
                    source=source,
                    verified_by="expert",
                ))
                print("  [OK] Edited and accepted.")
                break
            elif choice == "R":
                print("  [--] Rejected.")
                break
            elif choice == "?":
                print(f"\n  Full source text:\n  {teaching_text}\n")
            else:
                print("  Please enter A, E, R, or ?")

    print(f"\n{len(verified)}/{len(candidate_rules)} rules accepted into knowledge base.")
    return verified


# ---------------------------------------------------------------------------
# Knowledge base I/O
# ---------------------------------------------------------------------------

def kb_path(pair: ConfusablePair) -> Path:
    safe_name = (
        f"{pair.class_name_a.lower().replace(' ', '_')}"
        f"_vs_"
        f"{pair.class_name_b.lower().replace(' ', '_')}.json"
    )
    return KNOWLEDGE_BASE_DIR / safe_name


def save_rules(pair: ConfusablePair, rules: list[dict]) -> Path:
    path = kb_path(pair)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"pair": f"{pair.class_name_a} vs {pair.class_name_b}", "rules": rules}, f, indent=2, ensure_ascii=False)
    print(f"Knowledge base saved -> {path}")
    return path


def load_rules(pair: ConfusablePair) -> list[dict]:
    path = kb_path(pair)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rules", [])


def rules_to_prompt(rules: list[dict]) -> str:
    """Format verified rules as a numbered list for injection into GPT-4V prompt."""
    if not rules:
        return ""
    lines = ["Expert identification rules (verified):"]
    for i, r in enumerate(rules, 1):
        conf_tag = f"[{r.get('confidence', 'medium').upper()}]"
        lines.append(f"  {i}. {conf_tag} {r['rule']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main teaching session entry point
# ---------------------------------------------------------------------------

def run_teaching_session(
    pair: ConfusablePair,
    auto_accept: bool = False,
) -> list[dict]:
    """
    Full teaching session for one species pair.

    Reads the teaching text from teaching_sessions/<pair.teaching_file>,
    extracts rules via GPT-4, verifies interactively (or auto-accepts if
    auto_accept=True, for non-interactive use in testing).

    Returns the list of verified rules (also saved to knowledge_base/).
    """
    teaching_path = TEACHING_DIR / pair.teaching_file
    if not teaching_path.exists():
        raise FileNotFoundError(
            f"Teaching text not found: {teaching_path}\n"
            f"Please add field-guide text for this pair."
        )

    teaching_text = teaching_path.read_text(encoding="utf-8").strip()
    source = f"teaching_sessions/{pair.teaching_file}"

    client = OpenAI(api_key=get_openai_key())
    candidates = extract_rules(client, pair, teaching_text)

    if auto_accept:
        rules = [
            _make_rule(
                pair=pair,
                rule_text=r["rule"],
                feature=r.get("feature", ""),
                favors=r.get("favors", ""),
                confidence=r.get("confidence", "medium"),
                source=source,
                verified_by="auto",
            )
            for r in candidates
        ]
    else:
        rules = verify_rules_interactive(pair, candidates, teaching_text, source)

    save_rules(pair, rules)
    return rules


if __name__ == "__main__":
    # Quick smoke test — run teaching session for first pair
    from confusable_pairs import CONFUSABLE_PAIRS
    pair = CONFUSABLE_PAIRS[0]
    print(f"Teaching session for: {pair.class_name_a} vs {pair.class_name_b}")
    rules = run_teaching_session(pair, auto_accept=False)
    print(f"\nFinal rules in knowledge base: {len(rules)}")
