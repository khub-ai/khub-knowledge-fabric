"""
proposer_backends.py — Concrete Proposer implementations.

Two backends implement the abstract `Proposer` interface from
`proposer_schema.py`:

  FrontierProposer    Anthropic Claude via the messages + tool-use API.
                      Tool schemas are derived from the Pydantic response
                      models, so the model can only return a payload that
                      validates against the wire schema.
                      prior_weight = 0.6 (high trust per DSL §8.3).

  ConstrainedProposer OpenAI-compatible chat-completions endpoint serving
                      a small VLM (Qwen3-VL-8B etc.) via llama.cpp / vLLM /
                      Together.ai. JSON schema is requested via the
                      `response_format` field where supported, with a
                      Python-side validation fallback.
                      prior_weight = 0.3 (low trust per DSL §8.3).

Both backends are network-only — no model files are loaded in-process.
Both fail closed: any validation error or transport failure returns a
schema-valid empty/UNKNOWN response so the symbolic core can continue
on its own evidence.

Image delivery: when `image_png` is provided, both backends attach it as
a base64 vision content block alongside the symbolic JSON payload.

This module deliberately does NOT depend on the anthropic or openai SDKs
at import time — they are imported inside the backend constructors so
that the rest of the symbolic core can be exercised without those
packages installed (the test suite uses MockProposer).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any, Optional

from proposer_schema import (
    BindRolesRequest,
    BindRolesResponse,
    ExplainStallRequest,
    ExplainStallResponse,
    ProposalRequest,
    ProposalResponse,
    Proposer,
    ProposeActionModelRequest,
    ProposeActionModelResponse,
    ProposePreconditionRequest,
    ProposePreconditionResponse,
    RankGoalsRequest,
    RankGoalsResponse,
    RequestKind,
    EffectType,
    PreconditionType,
)


# =============================================================================
# Shared helpers
# =============================================================================

# Map RequestKind → response model class. Used by both backends to validate
# JSON returned by the LLM.
_RESPONSE_FOR_REQUEST = {
    RequestKind.BIND_ROLES:            BindRolesResponse,
    RequestKind.RANK_GOALS:            RankGoalsResponse,
    RequestKind.PROPOSE_ACTION_MODEL:  ProposeActionModelResponse,
    RequestKind.PROPOSE_PRECONDITION:  ProposePreconditionResponse,
    RequestKind.EXPLAIN_STALL:         ExplainStallResponse,
}


def _empty_response(request: ProposalRequest) -> ProposalResponse:
    """Schema-valid no-op response for a given request kind. Used as the
    fail-closed fallback when the backend errors or returns invalid JSON."""
    rk = request.request_kind
    if rk == RequestKind.BIND_ROLES:
        return BindRolesResponse(bindings=[], rationale="fallback: no proposer output")
    if rk == RequestKind.RANK_GOALS:
        return RankGoalsResponse(rankings=[], rationale="fallback: no proposer output")
    if rk == RequestKind.PROPOSE_ACTION_MODEL:
        return ProposeActionModelResponse(
            action_id=request.action_id,
            effect_type=EffectType.NO_OP,
            precondition=PreconditionType.ALWAYS,
            rationale="fallback: no proposer output",
        )
    if rk == RequestKind.PROPOSE_PRECONDITION:
        return ProposePreconditionResponse(
            action_id=request.action_id,
            precondition=PreconditionType.ALWAYS,
            rationale="fallback: no proposer output",
        )
    if rk == RequestKind.EXPLAIN_STALL:
        first = request.context.available_actions[0] \
            if request.context.available_actions else None
        return ExplainStallResponse(
            suggested_action_id=first,
            rationale="fallback: no proposer output",
        )
    raise ValueError(f"unknown request kind: {rk}")


def _system_prompt() -> str:
    return (
        "You are the Proposer for a symbolic ARC-AGI-3 game agent. "
        "You receive a symbolic snapshot of the current game state plus "
        "an optional rendered frame. You must reply ONLY with a JSON "
        "object matching the requested schema. No prose outside the "
        "JSON. Use the UNKNOWN role and short rationales when uncertain "
        "rather than guessing."
    )


def _user_payload(request: ProposalRequest) -> str:
    """Serialize the request to a compact JSON string for the LLM."""
    return request.model_dump_json(exclude_none=True)


def _validate_response(
    request: ProposalRequest, raw_json: str
) -> Optional[ProposalResponse]:
    """Best-effort parse + Pydantic validation. Returns None on failure."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    cls = _RESPONSE_FOR_REQUEST.get(request.request_kind)
    if cls is None:
        return None
    # Stamp the discriminator if the model omitted it.
    data.setdefault("request_kind", request.request_kind.value)
    try:
        return cls.model_validate(data)
    except Exception:
        return None


def _b64_png(image_png: bytes | None) -> Optional[str]:
    if not image_png:
        return None
    return base64.standard_b64encode(image_png).decode("ascii")


# =============================================================================
# Frontier backend — Anthropic Claude
# =============================================================================

class FrontierProposer(Proposer):
    """Claude-backed Proposer for the arcprize.com leaderboard run.

    Network-only; uses the Messages API with a single tool whose
    input_schema is generated from the response Pydantic model. This
    forces structured output without relying on prose parsing.
    """

    mode = "frontier"
    prior_weight = 0.6

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "FrontierProposer requires the `anthropic` package"
            ) from e
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self._model = model
        self._max_tokens = max_tokens

    # ----------------------------------------------------------------------

    async def propose(
        self,
        request: ProposalRequest,
        image_png: bytes | None = None,
    ) -> ProposalResponse:
        cls = _RESPONSE_FOR_REQUEST[request.request_kind]
        tool_name = f"emit_{request.request_kind.value}"
        tool = {
            "name": tool_name,
            "description": f"Emit a {cls.__name__} payload.",
            "input_schema": cls.model_json_schema(),
        }

        content: list[dict[str, Any]] = []
        b64 = _b64_png(image_png)
        if b64:
            content.append({
                "type": "image",
                "source": {"type": "base64",
                           "media_type": "image/png",
                           "data": b64},
            })
        content.append({"type": "text", "text": _user_payload(request)})

        try:
            msg = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=_system_prompt(),
                tools=[tool],
                tool_choice={"type": "tool", "name": tool_name},
                messages=[{"role": "user", "content": content}],
            )
        except Exception:
            return _empty_response(request)

        # Extract the tool_use block; that's our structured response.
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                payload = json.dumps(block.input)
                resp = _validate_response(request, payload)
                if resp is not None:
                    return resp
                break
        return _empty_response(request)


# =============================================================================
# Constrained backend — small open VLM via OpenAI-compatible API
# =============================================================================

class ConstrainedProposer(Proposer):
    """Open-VLM-backed Proposer for the air-gapped Kaggle competition run.

    Speaks the OpenAI Chat Completions protocol. Works with llama.cpp
    server, vLLM, Together.ai, or any other endpoint exposing the same
    surface. JSON Schema enforcement is requested via `response_format`
    when the server supports it; otherwise we fall back to a strict
    Python-side parse.
    """

    mode = "constrained"
    prior_weight = 0.3

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        base_url: str = "http://127.0.0.1:8080/v1",
        api_key: Optional[str] = None,
        max_tokens: int = 512,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise RuntimeError(
                "ConstrainedProposer requires the `openai` package"
            ) from e
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
        )
        self._model = model
        self._max_tokens = max_tokens

    # ----------------------------------------------------------------------

    async def propose(
        self,
        request: ProposalRequest,
        image_png: bytes | None = None,
    ) -> ProposalResponse:
        cls = _RESPONSE_FOR_REQUEST[request.request_kind]
        schema = cls.model_json_schema()

        user_content: list[dict[str, Any]] = []
        b64 = _b64_png(image_png)
        if b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        user_content.append({
            "type": "text",
            "text": (
                f"Respond with a single JSON object matching this schema:\n"
                f"{json.dumps(schema)}\n\n"
                f"Request payload:\n{_user_payload(request)}"
            ),
        })

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": user_content},
            ],
        }
        # Try strict JSON-schema mode first; some servers ignore this.
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": cls.__name__, "schema": schema, "strict": True},
        }

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception:
            # Retry without strict response_format — some endpoints reject it.
            kwargs.pop("response_format", None)
            try:
                resp = await self._client.chat.completions.create(**kwargs)
            except Exception:
                return _empty_response(request)

        try:
            text = resp.choices[0].message.content or ""
        except Exception:
            return _empty_response(request)

        # The model may wrap JSON in fences or include leading prose.
        text = _extract_json_object(text)
        validated = _validate_response(request, text)
        return validated if validated is not None else _empty_response(request)


# =============================================================================
# JSON extraction (small VLMs love to wrap their output in ``` fences)
# =============================================================================

def _extract_json_object(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        # ```json\n...\n```  or  ```\n...\n```
        s = s.strip("`")
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        if s.endswith("```"):
            s = s[:-3]
    # Find first { and matching final }
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        return s
    return s[start:end + 1]


# =============================================================================
# Self-test (no network calls — exercises validation + fallback paths only)
# =============================================================================

if __name__ == "__main__":
    from proposer_schema import (
        ProposalContext,
        ObjectSummary,
    )

    ctx = ProposalContext(
        frame_shape=(10, 10),
        available_actions=["ACTION1", "ACTION2"],
        objects=[ObjectSummary(id=1, color=8, area=1, centroid=(4.0, 4.0),
                               bbox=(4, 4, 4, 4))],
    )
    req = BindRolesRequest(context=ctx)

    # 1. _empty_response always validates.
    print("empty bind:", _empty_response(req))

    # 2. _validate_response on garbage → None.
    assert _validate_response(req, "not json") is None

    # 3. _validate_response on a real payload → object.
    good = '{"bindings": [], "rationale": "ok"}'
    parsed = _validate_response(req, good)
    assert parsed is not None and parsed.rationale == "ok"

    # 4. _extract_json_object handles fences.
    fenced = "```json\n{\"a\": 1}\n```"
    assert _extract_json_object(fenced).strip() == '{"a": 1}'

    print("self-test OK")
