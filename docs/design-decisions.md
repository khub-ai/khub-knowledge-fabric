# Design Decisions

This document explains the key design choices behind the project, how it differs from existing approaches, and the known limitations of the current implementation.

## How this differs from existing agent memory

Most AI assistants in 2026 offer some form of persistent memory. Here is how this project differs:

| Feature | Context-window memory | Platform memory (e.g., ChatGPT Memory, Claude Projects) | This project (PIL) |
|---|---|---|---|
| **Storage location** | Server-side | Server-side | Local, user-owned |
| **User can inspect** | Limited | Partial (some platforms show stored memories) | Full — artifacts are text files |
| **User can edit** | No | Limited | Yes — any text editor |
| **User can delete** | Session reset only | Per-item deletion in some platforms | Full control (file system) |
| **Portable** | No | No | Yes — model-agnostic text |
| **Structured** | No (raw conversation) | Minimal (key-value or short statements) | Yes — typed, versioned, with provenance |
| **Generalized** | No | Rarely | Yes — the pipeline explicitly generalizes observations into rules |
| **Governable** | No | Platform-dependent | Yes — user owns the files, controls what persists |

The key difference is not just *where* the knowledge is stored, but *what form* it takes. Platform memory features typically store brief statements extracted from conversation ("user prefers dark mode"). PIL produces structured, typed, confidence-scored, versioned artifacts that the user can reason about, edit, and share.

## Artifact format: free-form text, not rigid schemas

Knowledge artifacts are primarily **free-form text with lightweight conventions** — not rigid database schemas. This is a deliberate choice with several motivations:

- **Human-readable**: A user can open an artifact file and understand it immediately without specialised tooling.
- **Model-agnostic**: Any sufficiently capable LLM can consume the artifacts, regardless of vendor. There are no embeddings, token IDs, or model-specific representations.
- **Forward-compatible**: As the system evolves and new knowledge types emerge, existing artifacts don't require migration. The format accommodates what we haven't anticipated.
- **Editable**: Users can modify artifacts with any text editor, version-control them with git, diff them, and merge them.

The "schema" is a set of conventions — kind, confidence, provenance, timestamps — rather than a strict type system. This keeps artifacts open-ended enough to support diverse knowledge types while providing enough structure for the pipeline to operate on them programmatically.

## Model-agnostic: what it means precisely

"Model-agnostic" applies to two different layers:

1. **The artifacts are fully model-agnostic.** They are free-form text. Any LLM that can read and reason over text can consume them. There are no vendor-specific embeddings, fine-tuning artifacts, or proprietary formats. A user can export artifacts from an OpenClaw instance running Claude and import them into one running GPT, Gemini, or an open-source model.

2. **The PIL pipeline requires a capable host.** The code that elicits, induces, validates, and applies knowledge relies on the host agent having sufficient interactive learning and reasoning ability. In practice, this means leading LLMs from major vendors work well, but smaller or older models may not have the reasoning capacity to perform effective induction or generalization.

This distinction is important: the knowledge is portable even if the pipeline that produced it is not universally deployable.

## Confidence gating: current approach and intended evolution

The current implementation uses **word-level heuristics** (hedging words lower confidence, assertive words raise it) to set artifact confidence scores. This is a known simplification — a placeholder, not the final design.

The intended evolution:

1. **Phase 1 (current):** Heuristic confidence based on linguistic signals. Functional but not calibrated against outcomes.
2. **Phase 2:** Confidence calibrated from user feedback. When the agent applies an artifact and the user accepts, rejects, or edits the result, the confidence score adjusts. Over time, artifacts that lead to good outcomes gain confidence; those that don't, lose it.
3. **Longer-term:** Confidence may incorporate domain-specific signals, cross-artifact consistency checks, and temporal decay for time-sensitive knowledge.

The confidence-gating mechanism itself — suggest below threshold, auto-apply above — is designed to be pluggable. The threshold, the scoring method, and the feedback loop can all evolve without changing the artifact format or storage model.

## Security and governance considerations

This project stores knowledge as local files owned by the user. This design choice has security implications that are worth acknowledging:

**Current scope (Phase 1):** Knowledge ownership is simple — whoever owns the OpenClaw instance that created the knowledge owns it. Artifacts are local files with the same access controls as any other file on the user's machine.

**Known considerations for future phases:**

- **PII in artifacts**: The agent may learn facts that contain personally identifiable information. The current design stores these as plain text files. Future phases should consider consent controls per artifact kind and the ability to mark artifacts as sensitive.
- **Knowledge export**: If artifacts can be exported and shared, there must be controls to prevent accidental export of sensitive knowledge. This is a Phase 4/5 concern.
- **Enterprise deployment**: In an org context, questions arise about who can see, edit, or delete knowledge artifacts, and whether certain knowledge should be org-owned vs. individual-owned. This is deliberately deferred to Phase 5.

These are acknowledged as open design questions, not solved problems. The current foundation is designed to be extensible for governance without requiring fundamental changes to the artifact format.

## Why an OpenClaw extension

This project is built as an extension for [OpenClaw](https://github.com/openclaw/openclaw) for several reasons:

- **Plugin architecture**: OpenClaw has a mature plugin SDK that allows extensions to register tools, CLI commands, and hooks without forking upstream.
- **Multi-channel**: OpenClaw connects to 24+ messaging platforms, which means knowledge learned in one channel is available across all channels.
- **Local-first**: OpenClaw runs on the user's own machine, which aligns with the project's principle that knowledge should be user-owned and local.
- **Additive**: Users can install and upgrade OpenClaw normally. This extension adds capabilities without modifying the core.

The knowledge artifact format itself is not OpenClaw-specific. The PIL pipeline is implemented as an OpenClaw plugin for convenience, but the artifacts it produces are portable to any system that can read text files.
