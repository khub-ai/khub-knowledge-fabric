# Example: Learning in Action

This example shows how an agent progressively learns from interaction — observing patterns, generalizing them into reusable knowledge, and optionally compiling procedures into programs. It demonstrates the transition across memory types: episodic observation → semantic convention → procedural recipe → executable program.

## The scenario: managing financial statements

```
── Session 1 ──────────────────────────────────────────────────

User: Download my January statement from Chase, rename it
      "2026-01-chase-checking.pdf", and put it in my
      Finance/Statements folder.

Agent: Done. Statement saved to ~/Finance/Statements/2026-01-chase-checking.pdf.

── Session 2 ──────────────────────────────────────────────────

User: Download my February statement from Chase. Same as last time.

Agent: Got it — renamed to "2026-02-chase-checking.pdf" and saved
       to ~/Finance/Statements/.

       I notice you follow a consistent naming pattern:
       YYYY-MM-institution-account.pdf. Want me to remember this
       for future downloads?

User: Yes, always use that pattern.

       ⟹ [Semantic artifact created]
       kind: convention
       content: "Financial statements are named
                 YYYY-MM-institution-account.pdf
                 and stored in ~/Finance/Statements/"
       confidence: 0.85

── Session 3 ──────────────────────────────────────────────────

User: Get my February statement from Fidelity too.

Agent: Saved to ~/Finance/Statements/2026-02-fidelity-brokerage.pdf.

       By the way — you now download statements from two institutions.
       Want me to create a monthly checklist so we don't miss any?

User: Good idea. Add Amex too — I always forget that one.

       ⟹ [Semantic artifact created]
       kind: fact
       content: "User's financial institutions: Chase (checking),
                 Fidelity (brokerage), Amex (credit card)"

       ⟹ [Procedural artifact created — structured recipe]
       kind: procedure
       content: "Monthly statement download:
                 1. For each institution [Chase, Fidelity, Amex]:
                    a. Log in and download the latest statement PDF
                    b. Rename to YYYY-MM-institution-account.pdf
                    c. Save to ~/Finance/Statements/
                 2. Confirm all files are present
                 Frequency: once per month, after the 1st"
       confidence: 0.75

── Session 4 ──────────────────────────────────────────────────

User: It's March 2nd — time for statements.

Agent: On it. Following your monthly checklist:
       ✓ 2026-02-chase-checking.pdf
       ✓ 2026-02-fidelity-brokerage.pdf
       ✓ 2026-02-amex-credit.pdf
       All saved to ~/Finance/Statements/.

       I've now run this procedure three months in a row and the
       steps are identical each time. Want me to turn it into a
       script you can run (or schedule) without needing me?

User: Yes, that would be great.

Agent: Here's a script that automates the download and renaming.
       I'll save it to ~/Finance/Scripts/download-statements.sh.
       You can also schedule it as a cron job on the 2nd of each
       month.

       [Generates and saves executable script]

       ⟹ [Procedural artifact updated]
       The structured recipe is retained as documentation.
       An executable script is now linked as an optional
       optimised form.

── Session 5 ──────────────────────────────────────────────────

User: I closed my Amex account. And I opened a Schwab account.

Agent: Updated:
       - Removed Amex from the institution list and the download
         procedure
       - Added Schwab (what type of account?)

User: Brokerage.

Agent: Done. Updated the institution list, the procedure recipe,
       and the download script. Here's what changed:

       [Shows diff of the recipe and the script]

       ⟹ [Semantic artifact revised: institution list updated]
       ⟹ [Procedural artifact revised: recipe + script updated]
```

## What this demonstrates

| Session | What happens | Memory type | PIL stage |
|---|---|---|---|
| 1 | Agent executes a one-off task | Episodic (observation only) | — |
| 2 | Agent detects a naming pattern and proposes a convention | Episodic → **Semantic** generalization | Elicit → Induce → Validate → Persist |
| 3 | Agent accumulates facts and proposes a structured procedure | **Semantic** + **Procedural** creation | Elicit → Induce → Persist |
| 4 | Agent suggests compiling a repeated procedure into a script | **Procedural** recipe → program (optional) | Apply → Compact |
| 5 | Circumstances change; agent revises artifacts and code together | **Revision** across types | Revise |

### Key principles visible in this example

- **Episodic → semantic generalization:** The agent observes a repeated naming pattern and proposes a convention — it doesn't just remember what happened, it distills a rule.
- **Semantic accumulation:** The agent learns the institution list as a fact artifact, separate from the procedure that uses it.
- **Procedural recipe as primary artifact:** The agent proposes a structured checklist that remains human-readable and editable.
- **Recipe → program is optional:** Only after repeated identical execution does the agent suggest compiling to a script. It doesn't assume code is always the goal. The recipe is retained as documentation and fallback.
- **Coherent revision:** When circumstances change (closed account, new account), the agent updates semantic and procedural artifacts together, including the generated script.
- **User stays in control:** Every generalization and escalation — observation → convention → recipe → script — is proposed by the agent and confirmed by the user.

---

## See also

This example shows *passive* learning from ordinary use — the agent observes, generalizes, and learns incrementally without a structured teaching interaction. A complementary pattern is *active* learning through structured dialogue with a domain expert:

→ *[Expert-to-Agent Dialogic Learning](../specs/expert-to-agent-dialogic-learning.md)* — how an agent learns deep judgment, procedures, and boundaries through purposeful questioning of a human expert
