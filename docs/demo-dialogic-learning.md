# Demo: Expert-to-Agent Dialogic Learning

This walkthrough shows Phase 4 end-to-end: an expert teaches the agent an
investing screening rule across one session, the agent promotes it to the
knowledge store, and a second session retrieves it automatically.

The demo uses `pil-chat` — the standalone CLI chatbot included in the repo.
No OpenClaw installation is required.

> **Note on the transcript below:** The agent responses shown are
> representative of what the system produces — exact wording varies because
> question generation, gap assessment, and synthesis all depend on LLM calls.
> The flow (gap-by-gap progression → synthesis proposal → correction →
> promotion) is deterministic; the wording is not.

---

## What this demonstrates

Most AI systems forget everything between sessions. A few support explicit
memory commands ("remember that…"). This demo shows something different:
**structured knowledge elicitation through dialogue** — the agent asks targeted
questions to close specific epistemic gaps, proposes a synthesis, and stores
the result as a durable, retrievable knowledge artifact.

This is the knowledge-acquisition equivalent of a structured interview, not a
conversation. The agent knows exactly which questions to ask and when to stop.

### Current implementation scope

| Feature | Status |
|---|---|
| Gap-driven question ladder (5 gaps) | ✅ Implemented |
| LLM synthesis + correction parsing | ✅ Implemented |
| Session persistence + `committed` flag | ✅ Implemented |
| Artifact promotion with session provenance | ✅ Implemented |
| Cross-session retrieval via Phase 1 pipeline | ✅ Implemented |
| Communication-profile calibration | 🔲 Deferred — `DEFAULT_PROFILE` used |
| Gap inheritance from prior sessions | 🔲 Deferred — metadata tracked, not used |
| Expert redirect (multi-thread) | 🔲 Deferred — single candidate rule per session |

---

## Prerequisites

```bash
git clone https://github.com/khub-ai/khub-knowledge-fabric
cd khub-knowledge-fabric
pnpm install
export ANTHROPIC_API_KEY=sk-ant-...   # macOS/Linux/WSL
# $env:ANTHROPIC_API_KEY="sk-ant-..." # Windows PowerShell
```

Alternatively, if `pnpm` is not on your `PATH` (or blocked by execution policy
on Windows), run pil-chat directly with Node (v18+) and the ts-node ESM loader:

```bash
node --loader ts-node/esm/transpile-only --no-warnings \
  tools/pil-chat/index.ts -- --fresh --verbose
```

On Windows PowerShell in a single line:
```powershell
node --loader ts-node/esm/transpile-only --no-warnings tools/pil-chat/index.ts -- --fresh --verbose
```

If pnpm itself is blocked by execution policy, fix it once with:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Session 1 — Teaching the rule

Start pil-chat with a clean store so earlier artifacts don't interfere:

```bash
pnpm chat -- --fresh --verbose
```

### Start a teach session

```
You: /teach investing "screen survivability before analyzing upside"
```

The agent creates a session file, records the domain and objective, and
immediately asks the opening case-elicitation question. The gap bar does not
appear on this line — it appears after your first response.

```
[Teach mode] Session: <uuid>  domain: investing

Agent: Can you walk me through a specific recent situation where you applied
       this screening approach — ideally a real company or case you actually
       looked at?
```

### Answer the agent's questions

The agent works through five gap types in order. You don't need to answer in
any particular format — just respond naturally.

**Turn 1 — case-elicitation**
```
You: I was looking at an industrial distributor last year — specialty fasteners,
     niche market. The stock got cheap after a supply chain disruption. Before
     I looked at margin recovery potential at all, I asked whether they could
     survive a normal rough patch without external help.
```
```
  Gaps: [✓ case] [ process] [ boundary] [ exception] [ revision]

Agent: What was your actual reasoning sequence? What did you look at first,
       and what would have caused you to stop?
```

**Turn 2 — process-extraction**
```
You: First question: can they keep their core customers through the rough patch?
     Second: can they service debt without a restructuring? Third: does survival
     depend on several favorable things happening at once? If yes to the third
     one, I deprioritize immediately — stacked contingencies are a red flag.
```
```
  Gaps: [✓ case] [✓ process] [ boundary] [ exception] [ revision]

Agent: Where does this rule stop applying? Are there situations where you'd
       skip the survivability check and go straight to upside?
```

**Turn 3 — boundary**
```
You: If the distress is obviously temporary and the business model is intact —
     like a one-time inventory write-down — I might skip it. Also, debt alone
     isn't the issue. The real question is fragility under realistic stress, not
     leverage per se.
```
```
  Gaps: [✓ case] [✓ process] [✓ boundary] [ exception] [ revision]

Agent: Can you give me an example where you applied this rule but got it wrong,
       or a case where following it led to a mistake?
```

**Turn 4 — counterexample / failure case**
```
You: Yes — I passed on a niche manufacturer because of the debt load without
     really testing whether they could flex their cost structure. They could.
     The balance sheet looked fragile but operationally they were resilient.
     I was too mechanical about it.
```
```
  Gaps: [✓ case] [✓ process] [✓ boundary] [✓ exception] [ revision]

Agent: What new evidence or changed circumstances would cause you to revise
       or set aside this screening approach?
```

**Turn 5 — revision trigger**
```
You: If I found a consistent pattern of passing on businesses that turned out
     fine, I'd know I'm calibrating the fragility check too harshly. That
     would be the signal to revisit.
```
```
  Gaps: [✓ case] [✓ process] [✓ boundary] [✓ exception] [✓ revision]
```

All five gaps are now closed. The agent proposes a synthesis:

```
Agent: Based on everything you've shared, here's what I've learned:

       "Before analyzing upside in a distressed situation, first ask whether
       the business can survive a normal rough patch under believable
       assumptions. Check: (1) can they retain core customers, (2) can they
       service debt without restructuring, (3) does survival require multiple
       favorable events simultaneously — if yes, deprioritize. Note that debt
       alone is not disqualifying; the real test is operational resilience under
       realistic stress. This rule is most useful when distress appears to have
       a structural rather than temporary cause. Revisit the rule if you find
       a consistent pattern of passing on businesses that subsequently proved
       resilient."

       Does this capture it correctly? If anything is wrong or missing, just
       tell me and I'll revise it.
```

### Confirm or correct the synthesis

```
You: That's right, though I'd add — check for at least 3 years of positive
     free cash flow as a baseline before you even start the survivability
     check. If they haven't generated cash, the check is moot.
```

The agent parses the correction, incorporates it, and promotes the rule:

```
Agent: I've recorded that as the rule for "investing":

       "Before analyzing upside in a distressed situation, first confirm at
       least 3 years of positive free cash flow as a baseline — if that's
       absent, deprioritize immediately. If FCF is present, ask whether the
       business can survive a normal rough patch: retain core customers, service
       debt without restructuring, and avoid dependence on multiple favorable
       events occurring simultaneously. Debt alone is not disqualifying; the
       real test is operational resilience under realistic stress. Revisit if
       you find a consistent pattern of passing on businesses that proved
       resilient."

       It's been added to your knowledge store (provenance: session:<uuid>).
       Type /endteach to exit teach mode or continue teaching.

[Session complete. Rule promoted to store.]
```

### Exit teach mode

```
You: /endteach
[Teach mode ended]
  Session   : <uuid>
  Domain    : investing
  Objective : screen survivability before analyzing upside
  Stage     : complete
  Rules     : 1 developed,  1 promoted to store
```

---

## Verify the artifact was stored

The promoted rule is in `artifacts.jsonl` with confidence `0.80` and
`stage: "consolidated"` — values set explicitly by `promoteSession()`,
bypassing the default extraction seeding:

```bash
grep '"provenance"' ~/.openclaw/knowledge/artifacts.jsonl | grep "session:"
```

Expected output (prettified):
```json
{
  "id": "<uuid>",
  "kind": "procedure",
  "content": "Before analyzing upside in a distressed situation...",
  "confidence": 0.8,
  "stage": "consolidated",
  "provenance": "session:<session-uuid> — Elicited via dialogic learning. Objective: ...",
  "tags": ["investing"]
}
```

`confidence: 0.8` equals the default auto-apply threshold, so the artifact
injects as `[established]` in a fresh session (no decay). After sufficient
time without retrieval, decay will reduce effective confidence below the
threshold and the label will fall to `[suggestion]`.

The session file is also retained as an audit record:
```bash
ls ~/.openclaw/knowledge/sessions/
cat ~/.openclaw/knowledge/sessions/<uuid>.json | python3 -m json.tool | grep -E '"stage"|"committed"'
# "stage": "complete",
# "committed": true,
```

---

## Session 2 — Inherited knowledge

Start a new pil-chat session **without** `--fresh` so the store persists:

```bash
pnpm chat -- --verbose
```

Ask something that touches the domain without using any of the rule's keywords
explicitly:

```
You: I'm looking at a retailer that got cheap after a demand shock. Worth digging in?
```

The rule is retrieved and injected by the Phase 1 pipeline (the same
`before_prompt_build` hook used in normal sessions — there is no special Phase 4
retrieval path). With `--verbose`, you'll see the injection line:

```
  [PIL] → [established] [procedure] "Before analyzing upside in a distressed
           situation, first confirm at least 3 years of positive free cash flow..."
```

The agent's answer will reflect the survivability-first framing without being
told to use it.

---

## Resume teaching in a new session

Start teaching again in the same domain. The session records which prior sessions
and artifact IDs exist in this domain — shown in the header:

```bash
pnpm chat -- --verbose
/teach investing "when to use replacement cost valuation"
```

```
[Teach mode] Session: <new-uuid>  domain: investing
  Inherited 1 artifact(s) from 1 prior session(s)

Agent: Can you give me a specific example of a situation where you turned to
       replacement cost as the primary valuation anchor?
```

> **Note:** The "Inherited N artifact(s)" line confirms the prior session's
> artifacts are recorded in the new session's metadata. Automatic gap inheritance
> (skipping questions already answered by prior sessions) is not yet implemented
> — the question ladder starts fresh. This is a known v1 limitation, tracked
> for a future revision.

---

## What makes this different

| Approach | How it works | Limitation |
|---|---|---|
| Chat memory (most AI tools) | Store recent messages verbatim | Degrades with length; no structure |
| Explicit save commands ("remember that…") | User decides what to save | Burden on user; no gap-filling |
| RAG over documents | Retrieve chunks at query time | Read-only; doesn't learn from dialogue |
| Fine-tuning | Update model weights | Expensive; opaque; can't be inspected |
| **PIL Phase 4 (this)** | Agent asks structured questions, closes five epistemic gaps, synthesises a durable rule | — |

The five-gap consolidation criterion is the key: the agent doesn't store what
you say — it extracts what you *know*, in a form that generalises to future
cases. The rule produced is inspectable, editable, and portable.

---

## Known v1 gaps vs. full spec

| Gap | Full spec | v1 |
|---|---|---|
| Adaptive questioning style | Communication-profile calibration via 3-question pre-session | Hardcoded `DEFAULT_PROFILE` |
| Gap inheritance | Prior session answers skip already-closed gaps | Not implemented; metadata tracked only |
| Multi-thread teaching | Expert can redirect mid-session; paused rules resumable | Single active rule per session |
| Dashboard integration | Gap bar visible in web dashboard | CLI gap bar only |

---

## Reference

- [Spec: Expert-to-Agent Dialogic Learning](../specs/expert-to-agent-dialogic-learning.md)
- [Worked example (investing domain)](../specs/expert-to-agent-dialogic-learning-example-investing.md)
- [Positioning: what's unique about this approach](dialogic-learning-positioning.md)
- [Source: `src/session.ts`](../packages/knowledge-fabric/src/session.ts)
- [Source: `src/dialogue.ts`](../packages/knowledge-fabric/src/dialogue.ts)
