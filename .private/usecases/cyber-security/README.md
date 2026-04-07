# Knowledge Fabric Use Case: Cybersecurity Detection and Investigation Patching
## Turning Threat Intelligence and Analyst Judgment into Runtime Knowledge Artifacts

---

> **Status**: Design spec only — not yet implemented  
> **Theme**: Knowledge Fabric (KF) as a runtime knowledge layer for cybersecurity detection, triage, and investigation  
> **Last updated**: 2026.04.01  

[Knowledge Fabric (KF)](../../docs/what-is-kf.md) is a strong fit for cybersecurity because the domain already has abundant public expert knowledge, high commercial urgency, and a large gap between what analysts know and what deployed AI systems apply reliably. This use case proposes a cybersecurity specialization in which KF captures expert detection and investigation knowledge in human-readable [knowledge artifacts](../../docs/glossary.md#knowledge-artifact), applies them immediately at runtime, and revises them incrementally as new threats and failure modes appear.

---

## 1. The Problem

Cybersecurity teams already have several kinds of intelligence:

- threat reports that describe attacker behavior
- public detection logic such as Sigma rules
- ATT&CK mappings for tactics and techniques
- vendor playbooks and investigation guides
- internal analyst judgment about recurring false positives, edge cases, and escalation thresholds

But this knowledge is fragmented across documents, platforms, and teams. When an analyst discovers a useful triage heuristic or a better way to interpret a signal, that improvement is often lost in one of four ways:

- it stays in a senior analyst's head
- it is written as an informal note or ticket comment
- it becomes a brittle prompt tweak in an AI assistant
- it requires a slower engineering workflow before it changes operational behavior

KF proposes a different path: capture that expert knowledge in natural language, turn it into explicit reusable artifacts, apply it at runtime, and keep it reviewable, governable, and portable across underlying models and products.

---

## 2. Why Cybersecurity Is A Strong KF Domain

Cybersecurity is attractive as a KF use case for three reasons:

1. **The market is large and urgent**. Security is one of the most visible enterprise AI adoption areas because detection quality, analyst productivity, and response time have direct business impact.

2. **Public expert knowledge is abundant**. Unlike many specialized domains, cyber already has broad public expert material:
   - [MITRE ATT&CK](https://attack.mitre.org/)
   - [ATT&CK Enterprise tactics and techniques](https://attack.mitre.org/tactics/enterprise/)
   - [Sigma rules](https://sigmahq.io/sigma)
   - [Sigma rule repository](https://github.com/SigmaHQ/sigma)
   - [Splunk Analytic Stories](https://help.splunk.com/en/splunk-enterprise-security-7/user-guide/7.3/analytic-stories)
   - [CISA incident and vulnerability response playbooks](https://www.cisa.gov/ncas/current-activity/2021/11/16/new-federal-government-cybersecurity-incident-and-vulnerability)

3. **The KF value proposition is intuitive**. Analysts routinely discover localized improvements that do not justify model retraining but matter immediately in practice: better triage logic, better severity mapping, better evidence requirements, better escalation criteria, or better procedural response.

This makes cybersecurity a domain where KF can demonstrate something important: not that AI can read security documents, but that expert security knowledge can become immediately deployable runtime patches.

---

## 3. Recommended First Use Case

The strongest first cybersecurity use case is:

**CTI-to-detection-and-investigation patching**

In this design, KF ingests public threat intelligence and analyst guidance, then produces explicit artifacts that help:

- detect or prioritize relevant signals
- explain why a signal matters
- guide the next investigation steps
- revise logic after false positives or missed detections are discovered

This is a safer and more benchmarkable first target than generic SOC alert triage because public expert material is excellent, while public analyst-grade triage labels are relatively limited.

### Why this is the best starting point

- The expert knowledge is public and high quality.
- The task is operationally meaningful.
- The output can be evaluated in stages.
- The artifact can later be compiled into detection logic, investigation checklists, or tool-assisted workflows.

---

## 4. What KF Would Actually Learn

In this cyber use case, KF would learn several kinds of knowledge:

- **Detection heuristics**: what evidence pattern should suggest a given technique or threat
- **Triage criteria**: what should raise, lower, or defer analyst concern
- **Investigation checklists**: what to verify next, and in what order
- **Boundary conditions**: when a pattern is common but benign, incomplete, or not actionable
- **Procedural artifacts**: repeated investigative steps that can eventually be compiled into tools

Examples:

- "If suspicious PowerShell activity occurs but there is no encoded command, no network beaconing, and no credential access context, downgrade priority unless other persistence indicators are present."
- "If a script creates scheduled tasks and also modifies startup locations, escalate because the combination is more diagnostic than either signal alone."
- "If the observed behavior maps to ATT&CK T1059 but the parent process and signer are both expected in this environment, request enrichment before escalation rather than treating it as a high-confidence alert."

These are exactly the kinds of judgments analysts make repeatedly, and exactly the kinds of judgments KF should be able to preserve outside the model.

---

## 5. Candidate Experiment Structure

The cyber use case should be staged rather than attempted as one large benchmark.

### Phase 1: CTI -> explicit detection / investigation artifacts

**Task**:
- Input a public threat report, ATT&CK technique description, or analyst narrative.
- Ask KF to generate:
  - a structured detection hypothesis
  - an investigation checklist
  - explicit conditions that increase or decrease confidence

**Goal**:
- Show that KF can turn expert prose into explicit, reviewable, reusable security artifacts better than raw prompting or raw retrieval alone.

**Good public sources**:
- [MITRE ATT&CK](https://attack.mitre.org/)
- [Sigma rules](https://github.com/SigmaHQ/sigma)
- [Splunk Analytic Stories](https://help.splunk.com/en/splunk-enterprise-security-7/user-guide/7.3/analytic-stories)
- [CISA playbooks](https://www.cisa.gov/ncas/current-activity/2021/11/16/new-federal-government-cybersecurity-incident-and-vulnerability)
- [Microsoft CTI-REALM announcement](https://www.microsoft.com/en-us/security/blog/2026/03/20/cti-realm-a-new-benchmark-for-end-to-end-detection-rule-generation-with-ai-agents/)

### Phase 2: Telemetry-backed investigation support

**Task**:
- Present telemetry or event sequences from a public security dataset.
- Ask KF to apply stored artifacts to decide:
  - likely ATT&CK mapping
  - likely severity or confidence
  - recommended next investigation steps

**Candidate public datasets**:
- [LANL Comprehensive Cybersecurity Events](https://csr.lanl.gov/data/cyber1/)
- [LANL data index](https://csr.lanl.gov/data/)
- [CSE-CIC-IDS2018 overview](https://fkie-cad.github.io/COMIDDS/content/datasets/cse_cic_ids2018/)

**Goal**:
- Show that the same KF artifact can influence operational interpretation on new inputs.

### Phase 3: Tool generation for repetitive analyst work

**Task**:
- When a checklist step becomes repetitive and precise, let KF generate a small helper tool or query template.

**Goal**:
- Demonstrate that KF is not only a memory layer, but also a bridge from expert reasoning into executable support.

---

## 6. What Would Make KF Distinctive Here

Cybersecurity already has:

- detection rules
- knowledge bases
- retrieval systems
- playbooks
- SOAR automation
- LLM copilots

KF should not be pitched as merely another one of those.

The distinctive claim is this:

**KF gives cybersecurity teams a runtime layer that can learn, revise, govern, and deploy explicit expert knowledge patches in natural language, without waiting for retraining or deep product re-engineering.**

That means:

- an analyst can express a correction in plain language
- KF can turn it into a scoped artifact
- the artifact can change the next decision immediately
- the artifact remains reviewable and auditable
- the same artifact can later be refined, shared, deprecated, or compiled into a more formal tool

That full loop is rarely available in one system today.

---

## 7. Proposed KF Artifacts For Cybersecurity

The main artifact types for this use case would be:

- **Judgment artifacts**: severity or triage criteria
- **Boundary artifacts**: when not to over-escalate
- **Strategy artifacts**: investigation order and decision structure
- **Procedure artifacts**: repeatable response or enrichment sequences
- **Failure artifacts**: patterns that caused bad past judgments and should not be repeated

Example artifact shapes:

### Detection / triage artifact

```json
{
  "artifact_type": "judgment",
  "domain": "cyber-security",
  "scope": "powershell_suspicious_execution",
  "condition": "If PowerShell execution is present but there is no encoded command, no outbound beaconing, and the parent process is a known enterprise management tool",
  "action": "downgrade confidence and request host context before escalation",
  "rationale": "Single weak indicators are common in enterprise administration; escalation should depend on corroborating persistence or credential access evidence."
}
```

### Investigation checklist artifact

```json
{
  "artifact_type": "strategy",
  "domain": "cyber-security",
  "scope": "suspected_scheduled_task_persistence",
  "steps": [
    "Check whether the task creator is signed and expected in the environment",
    "Check for concurrent Run key or service creation activity",
    "Check whether the task launches encoded PowerShell or LOLBins",
    "Escalate if at least two persistence indicators co-occur"
  ]
}
```

---

## 8. Evaluation Plan

The experiment should make a narrow claim first.

### Primary claim

KF should beat simpler post-deployment alternatives on targeted cyber reasoning tasks by turning expert knowledge into reusable, explicit runtime artifacts.

### Baselines

- **Zero-shot LLM reasoning**
- **Few-shot prompting**
- **Raw retrieved reference text**
- **KF artifact pipeline**

### Metrics

- Artifact quality judged against expert source material
- Accuracy on targeted ATT&CK mapping or detection-support tasks
- Improvement on recurring edge cases after a patch is added
- Time-to-improvement after an analyst correction
- Reusability of the artifact across multiple examples
- Human auditability of the final reasoning chain

### Strong first success criterion

A convincing first result would be:

- KF performs better than zero-shot and raw retrieval on a bounded cyber task
- the improvement comes from an explicit reusable artifact
- the artifact is understandable to a security practitioner
- the improvement happens without fine-tuning the underlying model

---

## 9. Risks And Boundaries

This use case has several real risks:

- **Ground-truth scarcity**: public datasets often have attack labels, but not rich analyst triage labels
- **Artifact leakage risk**: if KF learns during evaluation, benchmark claims must be framed carefully
- **Environment dependence**: some cyber judgments depend on local context not visible in public data
- **Overclaim risk**: this should not be presented as production-ready SOC automation without partner data

Because of these risks, the README and any benchmark write-up should frame the first cyber experiment as:

- a demonstration of runtime expert patching in cybersecurity
- not a claim that KF can fully replace analysts or SIEM engineering

---

## 10. Implementation Sketch

Suggested repo structure:

```
usecases/
  cyber-security/
    README.md                 ← this design spec
    python/
      harness.py              ← task runner
      ensemble.py             ← KF round orchestration
      dataset.py              ← public cyber dataset loader
      rules.py                ← shim for RuleEngine
      tools.py                ← shim for ToolRegistry
      prompts/
        observer.txt
        mediator.txt
        verifier.txt
```

Suggested `dataset_tag`:

```text
cyber-security
```

Suggested first benchmark tasks:

1. CTI passage -> ATT&CK mapping + investigation artifact
2. Sigma / ATT&CK passage -> explicit triage checklist
3. Telemetry case -> apply stored artifact and decide likely interpretation

---

## 11. Why This Use Case Matters To Vendors

Cybersecurity vendors and enterprise security teams already live in a world of:

- post-deployment edge cases
- fast-changing adversary behavior
- noisy detections
- high cost of missed context

KF matters because it offers:

- faster knowledge deployment than model retraining
- explicit governance for sensitive detection logic
- portability across underlying models
- a path from analyst expertise to reusable product behavior
- a byproduct of labeled edge cases and explicit reasoning for later model improvement

This is one of the clearest enterprise settings where a **knowledge-release cycle** could matter more than a slower **model-release cycle**.

---

## 12. Recommended Next Step

The next practical step is not to build a full SOC copilot. It is to run one narrow, credible experiment:

**Pick 1-2 ATT&CK techniques and show that KF can turn public expert material plus analyst corrections into a reusable investigation artifact that improves a bounded evaluation task over zero-shot and raw retrieval.**

That will be enough to test:

- whether the cyber domain is a good fit for KF
- whether public expert material is sufficient
- whether the artifact story is strong and differentiated

If that works, telemetry-backed investigation support should be the next phase.
