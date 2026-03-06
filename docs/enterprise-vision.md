# Enterprise Vision

This document expands on the Phase 5 Governance & Ecosystem themes from the [roadmap](roadmap.md). It is written for two audiences: **enterprise adopters** evaluating whether this project addresses a real organizational need, and **investors** interested in the economic potential of portable knowledge artifacts.

---

## The enterprise problem this solves

### Tacit knowledge is the hardest knowledge to manage

Enterprise knowledge management has been a solved problem on paper for decades — wikis, document management systems, intranets, SharePoint. In practice it fails for one reason: the knowledge that matters most is not the kind that gets written down.

The knowledge that actually drives expert performance is *tacit*: the judgment about which client communication style to use, the mental model of which regulatory argument tends to hold up in practice, the hard-won understanding of which vendor claims can be trusted and in what contexts. This knowledge lives in people's heads because writing it down has historically required too much effort for too little return — documents don't surface themselves at the right moment, don't update automatically when circumstances change, and don't get read when they're needed.

AI agents change this equation. An agent that operates alongside a knowledge worker for months or years naturally encounters the contexts where tacit knowledge is exercised. With PIL, the agent can elicit that knowledge incrementally, structure it, and make it available — both to the same user in future sessions and, with appropriate governance, to others in the organization.

### The continuity problem

Consider what happens when a senior analyst leaves a law firm, a senior engineer leaves a product team, or an experienced underwriter leaves an insurance company:

- Their document output is preserved but their *judgment* — about which arguments work, which exceptions to flag, which edge cases to handle a certain way — is not.
- Onboarding the successor takes months not because the successor lacks capability but because they lack accumulated context.
- The team may not even know what was lost until it matters.

With PIL, a portion of that accumulated knowledge — the portions the employee and organization have agreed to retain — can be preserved as active artifacts. The successor's agent applies those artifacts from day one: surfacing relevant conventions, flagging known exceptions, and proposing handling patterns based on accumulated experience.

This is not a theoretical capability. It is the natural consequence of building knowledge management into the AI agent workflow itself, rather than relying on humans to document knowledge as a separate, unrewarded task.

---

## The enterprise deployment model

### Tiered knowledge stores

An enterprise PIL deployment would operate across four knowledge tiers:

```
Individual ──► Team ──► Org ──► Public
(private)   (reviewed)  (canonical)  (ecosystem)
```

Movement between tiers requires explicit human action:

- **Individual → Team**: The contributor flags an artifact as a team submission. A reviewer approves, rejects, or returns with comments.
- **Team → Org**: A publisher promotes reviewed artifacts to the org knowledge base. Policy controls can restrict which categories reach org level.
- **Org → Public**: An optional step for organizations that want to publish knowledge packages externally (e.g., a consultancy publishing domain-specific packages, or a professional body publishing regulatory guidance).

Within each tier, every organization's agents treat artifacts from higher tiers as authoritative context — injected alongside conversation history with appropriate confidence levels.

### Role-based access control

| Role | Permissions |
|---|---|
| **Contributor** | Create, edit, and delete own personal artifacts; submit artifacts to team review queue |
| **Reviewer** | View team submissions; approve, reject, or annotate; cannot promote directly to org level |
| **Publisher** | Promote team-reviewed artifacts to org level; set expiry dates and scope restrictions |
| **Auditor** | Read-only access to the full artifact registry across all tiers, including retired and superseded artifacts |
| **Admin** | Manage role assignments, configure retention policies, enforce category-level controls |

This maps cleanly onto existing enterprise identity management: roles can be assigned via SSO groups, and the artifact registry can be governed by the same access control policies as other internal systems.

### Compliance and auditability

Every artifact carries a structured provenance record from the moment it is created:

- **Creator**: which user, which agent instance
- **Source**: which conversation, which session
- **Confidence**: estimated at induction, updated by feedback signals
- **Revision history**: previous content, who revised it, when, and why
- **Retrieval log**: when the artifact was retrieved, for which conversation
- **Application log**: when the artifact was applied, whether the user accepted or overrode it
- **Retirement record**: when and why the artifact was retired, and by what it was superseded

This is not a separate logging system — it is the artifact's own lifecycle metadata, accumulated by the PIL pipeline as a natural consequence of operation.

For regulated industries, this matters concretely:

- **Legal**: You can demonstrate what knowledge the agent had at the time of a recommendation and who validated it.
- **Financial services**: You can trace model output back to the knowledge artifacts that informed it — relevant for explainability requirements under MiFID II, SEC rules, and similar frameworks.
- **Healthcare**: You can show that clinical guidance injected by an AI agent was derived from approved, reviewed knowledge, with a clear chain of human sign-off.
- **Any regulated context**: Artifact retirement creates a clean break — there is a record of when a rule changed and what replaced it, rather than a silent overwrite.

### Knowledge that survives the organization's evolution

Beyond employee turnover, there are structural events that routinely destroy organizational knowledge:

- **Mergers and acquisitions**: Two organizations with different conventions, processes, and judgment frameworks must somehow reconcile them.
- **System migrations**: Moving from one AI vendor or platform to another typically means losing the accumulated knowledge, since it is stored in proprietary formats.
- **Workforce scaling**: When a team grows rapidly, new members must somehow absorb conventions that were previously informal.

PIL's model addresses all three:

- **M&A**: Both knowledge bases can be exported, compared, and merged at the artifact level — conflicts are visible as contradicting artifacts and can be resolved by designated reviewers.
- **Platform migration**: Artifacts are model-agnostic text files. The same artifacts work on a new platform the day after migration, with no conversion step.
- **Scaling**: Org-level artifacts are injected into every new team member's agent from day one. The conventions are applied automatically without requiring explicit onboarding documentation.

---

## The investment thesis

### A new asset class: curated knowledge artifacts

Software infrastructure has repeatedly created value by defining standard formats that allow producers and consumers to exchange something of value without tight coordination:

- **npm** defined the standard package format for JavaScript. The value is not in npm's servers — it is in the coordination layer that allows any developer to publish a package and any other developer to consume it.
- **OpenAPI** defined the standard format for REST API descriptions. The value is in the interoperability it enables, not in any particular tool.
- **Docker images** defined the standard format for application containers. This made application distribution a solved problem.

PIL is positioned to define a standard format for **curated knowledge artifacts** — a new category of exchangeable, verifiable unit of expertise.

Once this standard achieves adoption, the economic surface is significant:

### Why portability creates a market

A knowledge artifact that is locked into a vendor's platform has no market — it cannot be extracted, traded, or valued independently. A knowledge artifact that is portable, verifiable, and model-agnostic is a product.

The same way open music formats (MP3) created the market for digital music distribution, and open video formats created the market for streaming, open knowledge artifact formats create the conditions for knowledge distribution markets.

### Potential business models on top of this layer

**Expert knowledge packages**

Domain experts with years of accumulated judgment — lawyers, accountants, engineers, physicians, consultants — can publish curated PIL packages representing their expertise. Consumers import these packages into their agents. The expert earns per-download, subscription, or licensing revenue.

This is analogous to selling a textbook, but the knowledge is *active* rather than inert: it surfaces at the right moment, adapts to context, and can be updated as the expert's knowledge evolves.

**Professional certification of knowledge packages**

For high-stakes domains, the value of a knowledge package is not just its content but its provenance. A package certified by a licensing body — reviewed by licensed practitioners, tested against case examples — commands a significant premium over an uncertified equivalent. This creates a certification business.

**Organizational knowledge custody**

Enterprises need the same things for knowledge packages that they need for software packages: private hosting, access controls, version management, audit trails. A managed private registry for org knowledge — with governance tooling layered on top — is a well-understood SaaS model (cf. Artifactory, GitHub Packages, AWS CodeArtifact) applied to a new asset class.

**Knowledge migration services**

Most organizations have large bodies of relevant knowledge trapped in document repositories, process manuals, email archives, and chat histories. Converting this into structured PIL artifacts is a one-time effort that pays recurring dividends as artifacts are injected into agent workflows. This is a services business initially, automatable over time.

**Knowledge continuity and M&A services**

Employee offboarding, team restructuring, and M&A create well-defined events where structured knowledge transfer has clear monetary value. A service that manages this transition — extracting, curating, and handing over knowledge artifacts — prices into the cost of each such event.

### Network effects and defensibility

The format standard is the defensible asset. As more organizations accumulate knowledge in PIL artifact format:

- Switching to a different format requires converting accumulated knowledge — an increasingly costly migration.
- The ecosystem of tools, converters, certification bodies, and marketplaces built around the standard makes the standard more valuable.
- Expert publishers who have built audiences for their knowledge packages have no reason to switch to an incompatible format.

This is the pattern of infrastructure standards: hard to establish initially, self-reinforcing once established, and very difficult to displace.

### What the technical foundation already provides

The artifact design choices made in phases 1–4 are not accidental — they are the minimum required to make Phase 5 possible without rearchitecting:

| Design choice | Why it matters for Phase 5 |
|---|---|
| Text-based artifacts | Human-readable, no vendor-specific serialisation, editable with standard tools |
| Model-agnostic | Knowledge packages are not invalidated by LLM vendor changes |
| Versioned with provenance | Audit trails and certification are a presentation layer, not a redesign |
| Additive-only field evolution | Phase 5 governance fields are backwards-compatible with Phase 1 artifacts |
| Local-first storage | No central dependency; orgs can host their own registry without a third-party service |
| Typed artifacts (semantic, procedural, evaluative) | Different governance rules can apply to different artifact kinds — evaluative judgments may require stricter review than factual conventions |

---

## Why now

Enterprise AI adoption is accelerating faster than enterprise AI governance. Most organizations deploying AI agents have no answer to the question "what does our agent know, and who approved it?" The compliance and audit requirements will follow — as they always do — once the technology is established enough to attract regulatory attention.

PIL is positioned to be the answer to that question: not a compliance bolt-on, but a knowledge management architecture where governance is a natural consequence of how knowledge is captured and stored, rather than an external constraint applied after the fact.

The window for establishing a standard format is open now, before proprietary formats lock in network effects. The technical groundwork is tractable. The enterprise need is real and growing. The economic surface — once the standard is established — is substantial.
