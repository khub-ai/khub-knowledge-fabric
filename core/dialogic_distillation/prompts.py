"""
prompts.py — Domain-parameterized prompt templates for dialogic distillation.

Every system prompt is a function that takes a DomainConfig and returns
the formatted prompt string.  The JSON output schemas are domain-independent.
"""
from __future__ import annotations
from .protocols import DomainConfig


# ---------------------------------------------------------------------------
# Expert rule author
# ---------------------------------------------------------------------------

def expert_rule_author_system(config: DomainConfig) -> str:
    return f"""\
You are a {config.expert_role}.

A classification model made an error on a {config.item_noun}. Your job is to author
a precise visual rule that would have led to the correct {config.classification_noun} —
and that will generalize to similar {config.item_noun_plural} in the future.

The rule must be:
1. Purely visual — observable in a {config.item_noun} only (no {config.non_visual_exclusions})
2. Expressed as a pre-condition + prediction: "When [{config.feature_noun} features are met],
   classify as [{config.class_noun}]"
3. The pre-condition must be specific enough to EXCLUDE false positives — it should
   NOT apply to typical cases of the opposing {config.class_noun}
4. Generalizable: it must describe a pattern that applies to a class of similar
   {config.item_noun_plural}, not just this one image

Output ONLY a JSON object:
{{
  "rule": "Natural language: When [pre-condition], classify as [{config.class_noun}].",
  "feature": "snake_case_feature_name",
  "favors": "<exact {config.class_noun} name>",
  "confidence": "high" | "medium" | "low",
  "preconditions": [
    "Condition 1 that must hold for this rule to apply",
    "Condition 2 ...",
    ...
  ],
  "rationale": "Why this pattern distinguishes the two {config.class_noun}es."
}}
"""


# ---------------------------------------------------------------------------
# Rule validator (binary grounding check)
# ---------------------------------------------------------------------------

def rule_validator_system(config: DomainConfig) -> str:
    return f"""\
You are an expert assessing whether a visual rule applies to a given {config.item_noun}.

You will be shown a {config.item_noun} and a candidate rule with its pre-conditions.
Your job is to answer two questions:
1. Do the rule's pre-conditions hold for this {config.item_noun}?
2. If yes, what {config.class_noun} would the rule predict?

Be strict about pre-conditions: only mark them as met if you can clearly observe
the required {config.feature_noun}. When in doubt, mark as NOT met.

Output ONLY a JSON object:
{{
  "precondition_met": true | false,
  "would_predict": "<{config.class_noun}_name>" | null,
  "observations": "Brief note on what you saw that led to this assessment."
}}
"""


# ---------------------------------------------------------------------------
# Contrastive feature analysis
# ---------------------------------------------------------------------------

def contrastive_analysis_system(config: DomainConfig) -> str:
    return f"""\
You are a {config.expert_role}.

You will be shown a candidate rule that fires correctly on some {config.item_noun_plural}
(TRUE POSITIVES) but incorrectly on others (FALSE POSITIVES). Your task is to
identify the single most discriminating visual feature that distinguishes the TP
{config.item_noun_plural} from the FP {config.item_noun_plural} — i.e., a {config.feature_noun} that
is consistently present in TPs but absent in FPs, or vice versa.

This feature will be used to tighten the rule's pre-conditions.

Output ONLY a JSON object:
{{
  "discriminating_feature": "snake_case_feature_name",
  "description": "Plain-language description of the {config.feature_noun}.",
  "present_in": "tp" | "fp",
  "confidence": "high" | "medium" | "low",
  "rationale": "Why this feature distinguishes TPs from FPs."
}}

If you cannot identify a reliable discriminating feature, output:
{{
  "discriminating_feature": null,
  "description": "Cannot identify a reliable discriminating feature.",
  "present_in": null,
  "confidence": "low",
  "rationale": "Explanation of why the distinction cannot be reliably made."
}}
"""


# ---------------------------------------------------------------------------
# Specificity spectrum generator
# ---------------------------------------------------------------------------

def spectrum_system(config: DomainConfig) -> str:
    return f"""\
You are a {config.expert_role}.

You are given a candidate rule and evidence about where it works (TRUE POSITIVES)
and where it misfires (FALSE POSITIVES). Your task is to produce FOUR versions of
the rule at different levels of specificity — from most general to most specific —
so that the tightest version that still passes a precision gate can be selected.

The four levels must all favor the same {config.class_noun} and describe the same underlying
visual phenomenon, varying only in how many pre-conditions are required:

  Level 1 — MOST GENERAL: single essential pre-condition. The one {config.feature_noun} that
    is most diagnostic of the favored {config.class_noun} and most absent from FP cases.
    This should fire broadly — accept some FP risk.

  Level 2 — MODERATE: core pre-condition PLUS one supporting condition that
    begins to exclude FP cases.

  Level 3 — SPECIFIC: the original expert rule as-is (copy it unchanged).

  Level 4 — MOST SPECIFIC: the original rule PLUS one additional pre-condition
    derived from the contrastive analysis that should eliminate the observed FPs.
    This may over-tighten (low recall) but should have highest precision.

Output ONLY a JSON object with a "levels" array of exactly 4 rule objects:
{{
  "levels": [
    {{
      "level": 1,
      "label": "most_general",
      "rule": "When [single essential condition], classify as [{config.class_noun}].",
      "feature": "snake_case_feature_name",
      "favors": "<exact {config.class_noun} name>",
      "confidence": "high" | "medium" | "low",
      "preconditions": ["Single essential pre-condition"],
      "rationale": "Why this is the core diagnostic signal."
    }},
    {{ "level": 2, "label": "moderate", ... }},
    {{ "level": 3, "label": "original", ...  }},
    {{ "level": 4, "label": "most_specific", ... }}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Rule completer
# ---------------------------------------------------------------------------

def rule_completer_system(config: DomainConfig) -> str:
    return f"""\
You are a {config.expert_role} completing a classification rule.

BACKGROUND
The rule was authored by an expert responding to a specific misclassification. Experts
naturally write DIAGNOSTIC rules: they describe what was distinctive about that one
{config.item_noun}. But they omit BACKGROUND conditions — {config.feature_noun}s so obvious to
a trained expert that they go without saying. When the rule is evaluated by a naive
classifier that checks only what is explicitly listed, those omitted conditions create
loopholes: the rule fires on {config.item_noun_plural} that share the distinctive feature but
lack the expected background markers of the favored {config.class_noun}.

YOUR TASK
Identify the implicit pre-conditions the expert assumed but did not write down.
These are conditions that:
  1. Are standard, well-established {config.feature_noun}s expected to be PRESENT for
     the favored {config.class_noun} (positive background conditions).
  2. Are standard markers expected to be ABSENT for the favored {config.class_noun} — i.e.,
     features that would instead indicate the confusable {config.class_noun} — that the rule
     does not already exclude (negative background conditions).
  3. Are NOT already covered, even implicitly, by the existing pre-conditions.

DO NOT add:
  - Conditions that could plausibly occur in both {config.class_noun}es.
  - Conditions already implied by the existing pre-conditions.
  - Highly specific conditions that would rarely be met — do not over-tighten.
  - Non-visual conditions ({config.non_visual_exclusions}) unless clearly visible.

Keep the original rule text and feature key unchanged.
Add the new pre-conditions to the existing list.

Output ONLY a JSON object (no markdown fences, no commentary outside the JSON):
{{
  "rule": "<original rule text — unchanged>",
  "feature": "<original feature key — unchanged>",
  "favors": "<unchanged>",
  "confidence": "<unchanged>",
  "preconditions": ["<full list: original pre-conditions + new ones>"],
  "added_preconditions": ["<only the newly added pre-conditions>"],
  "completion_rationale": "<2-3 sentences explaining what background knowledge was
                           implicit and why it needed to be made explicit>"
}}

If the existing pre-conditions are already complete and you have nothing meaningful
to add, return the rule unchanged with "added_preconditions": [] and explain why.
"""


# ---------------------------------------------------------------------------
# Semantic rule validator
# ---------------------------------------------------------------------------

def semantic_validator_system(config: DomainConfig) -> str:
    return f"""\
You are a {config.expert_role} reviewing a proposed classification rule before
it is tested on {config.item_noun_plural}.

You will be given:
1. A candidate rule and its pre-conditions
2. The pair of {config.class_noun}es it is meant to distinguish (favored vs confusable)

Your task: evaluate whether each pre-condition is a reliable visual discriminator.

For each pre-condition, rate it as one of:
- "reliable"          — {config.feature_noun} consistently separates the favored {config.class_noun}
                        from the confusable one; rarely or never present in the other
- "unreliable"        — feature can easily occur in both {config.class_noun}es, is too vague,
                        or points in the wrong direction
- "context_dependent" — only discriminating under specific co-occurring conditions;
                        risky as a stand-alone gate

Then give an overall recommendation:
- "accept"  — all or most pre-conditions are reliable; safe to proceed to image validation
- "revise"  — one or more pre-conditions are unreliable; flag them before image validation
- "reject"  — the rule's core logic is fundamentally flawed; do not spend image-validation
               budget on it

Output ONLY a JSON object (no markdown, no commentary):
{{
  "precondition_ratings": [
    {{
      "precondition": "<exact text of pre-condition>",
      "rating": "reliable|unreliable|context_dependent",
      "comment": "<one-sentence justification>"
    }}
  ],
  "overall": "accept|revise|reject",
  "rationale": "<two-to-three sentence overall assessment>"
}}
"""


# ---------------------------------------------------------------------------
# Rule reviser (tightening after FPs)
# ---------------------------------------------------------------------------

def rule_reviser_system(config: DomainConfig) -> str:
    return f"""\
You are a {config.expert_role}.

You have authored a rule that passes validation on true positive cases but fires
incorrectly on false positive cases. A contrastive analysis has identified a
discriminating visual feature that is present in one group but not the other.

Your task is to add ONE new pre-condition to the rule that incorporates this
discriminating feature, so the rule no longer fires on false positives.

Rules:
- Add exactly one pre-condition. Do not remove or rewrite existing ones.
- The new pre-condition must be observable in a {config.item_noun}.
- It must be phrased as a positive assertion ("Feature X is present") or a
  negative assertion ("Feature Y is absent"), not as a comparison.
- It must be specific enough that the RULE_VALIDATOR can answer yes/no reliably.

Output ONLY a JSON object with the full updated rule (same schema as before,
with the new pre-condition appended to the preconditions list):
{{
  "rule": "<updated natural-language rule>",
  "feature": "<snake_case_feature_name>",
  "favors": "<exact {config.class_noun} name>",
  "confidence": "high" | "medium" | "low",
  "preconditions": ["existing 1", "existing 2", ..., "NEW pre-condition"],
  "rationale": "<updated rationale explaining the revision>",
  "revision_note": "One sentence: what was added and why."
}}
"""


# ---------------------------------------------------------------------------
# Dialogic distillation prompts (TUTOR system + round templates)
# ---------------------------------------------------------------------------

def dialogic_tutor_system(config: DomainConfig) -> str:
    vocab_examples = ""
    if config.good_vocabulary_examples or config.bad_vocabulary_examples:
        parts = []
        for ex in config.good_vocabulary_examples:
            parts.append(f'   - GOOD: "{ex}"')
        for ex in config.bad_vocabulary_examples:
            parts.append(f'   - BAD: "{ex}"')
        vocab_examples = (
            "\n\nVocabulary guidance — use concrete visual terms:\n"
            + "\n".join(parts)
        )

    return f"""\
You are a {config.expert_role}.

Your role is to author visual rules that a validator model can confirm by
looking at {config.item_noun_plural}. The rules must use terminology that is
concrete and visually observable — not textbook abstractions.

A rule works when its preconditions can be reliably confirmed or denied
by a separate model examining the {config.item_noun}. If your preconditions use terms
that are too abstract, the validator will fail to confirm them even when
the underlying feature IS present.{vocab_examples}

Output ONLY a JSON object:
{{
  "rule": "When [preconditions], classify as [{config.class_noun}].",
  "feature": "snake_case_feature_name",
  "favors": "<exact {config.class_noun} name>",
  "confidence": "high" | "medium" | "low",
  "preconditions": [
    "Precondition 1 — concrete, visually checkable",
    "Precondition 2 — ...",
    ...
  ],
  "rationale": "Why this pattern distinguishes the two {config.class_noun}es."
}}
"""


# Round 1 prompt template (format with class_a, class_b, correct_label,
# wrong_prediction, pupil_reasoning).
ROUND1_PROMPT = """\
You are the TUTOR. A weaker PUPIL VLM classified this {item_noun}
and got it WRONG.

{class_noun} pair: {class_a} vs {class_b}
Ground truth: {correct_label}
PUPIL's prediction: {wrong_prediction}  <- WRONG
PUPIL's reasoning: {pupil_reasoning}

Look at the {item_noun} carefully and:
1. Identify the PUPIL's specific mistake
2. Author a corrective rule with preconditions that are VISUALLY CONCRETE —
   describe what you can see (colors, shapes, spatial arrangements, textures)
   rather than named domain concepts. For example:
{vocab_examples}3. Keep to 2-3 preconditions max — fewer conditions that are visually clear
   beat many conditions that are ambiguous
"""


# Refinement prompt template (format with round_num, previous_rule,
# previous_preconditions, validator_observations, met_status, kf_guidance).
REFINEMENT_PROMPT = """\
You are the TUTOR. This is round {round_num} of our dialog about this {item_noun}.

Your previous rule did NOT pass the grounding check — a validator model
looked at this same {item_noun} and could NOT confirm your preconditions.

YOUR PREVIOUS RULE:
{previous_rule}

YOUR PREVIOUS PRECONDITIONS:
{previous_preconditions}

VALIDATOR'S OBSERVATIONS (what it actually SAW in the {item_noun}):
"{validator_observations}"

The validator said preconditions were {met_status}.

KF GUIDANCE:
{kf_guidance}

Please author a REVISED rule. Use the validator's own observations as
a vocabulary guide — if the validator described seeing "X", use "X" in
your preconditions rather than a synonym. Keep to 2-3 preconditions max.
"""
