"""
ensemble.py — Main orchestrator for the ARC-AGI debate ensemble.

New architecture: reasoning separated from execution.

Protocol:
  Round 0:  Rule matching — evaluate active rules against puzzle
  Round 1:  Solver(s) propose TEXT-ONLY hypotheses (parallel if multiple)
  Round 2:  MEDIATOR synthesizes hypotheses into pseudo-code
  Round 3:  EXECUTOR runs pseudo-code against all demo pairs (deterministic)
            if all pass -> apply to test input -> done
            if fail -> MEDIATOR revises (up to MAX_REVISIONS times)
  Final:    MEDIATOR updates rules based on outcome
"""

from __future__ import annotations
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Failed-hypothesis sidecar — persists solver hypotheses from failed runs
# so correction events (--insight) can reference what was actually wrong.
# ---------------------------------------------------------------------------
_FAILED_HYP_PATH = Path(__file__).parent / "failed_hypotheses.json"


def _load_failed_hypotheses() -> dict:
    try:
        return json.loads(_FAILED_HYP_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_failed_hypothesis(task_id: str, hypotheses: list[str]) -> None:
    data = _load_failed_hypotheses()
    data[task_id] = hypotheses
    tmp = _FAILED_HYP_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, _FAILED_HYP_PATH)


def _clear_failed_hypothesis(task_id: str) -> None:
    data = _load_failed_hypotheses()
    if task_id in data:
        data.pop(task_id)
        tmp = _FAILED_HYP_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, _FAILED_HYP_PATH)

from grid_tools import Grid, grids_equal, summarize
from metadata import TaskMetadata, SolverEntry, MediatorDecision, compute_outcome, print_task_summary
from agents import (
    run_solvers_round1, run_mediator_synthesize, run_mediator_revise,
    run_mediator_extract_preference, run_generalization_pass,
    run_tool_generator, run_tool_generator_fix, call_agent,
    format_task_for_prompt, DEFAULT_MODEL, DEFAULT_SOLVERS,
    reset_cost_tracker, get_cost_tracker,
)
from executor import (
    run_executor, ExecutionResult, format_execution_trace, tool_signatures,
    parse_new_tools, register_dynamic_tool, test_tool_code,
)
from rules import RuleEngine, RuleMatch
from tools import ToolRegistry
import display as disp

MAX_REVISIONS = 5  # how many times MEDIATOR can revise pseudo-code after failure


_TOOL_FIX_ATTEMPTS = 5   # max self-correction retries per tool


async def _generate_and_register_tools(
    mediator_text: str,
    human_in_loop: bool,
    log_fn,
    task: dict | None = None,
    tool_registry: ToolRegistry | None = None,
) -> list[dict]:
    """
    Parse new tool requests from MEDIATOR response, generate Python code for each,
    verify against demo pairs, and self-correct up to _TOOL_FIX_ATTEMPTS times.

    If tool_registry is provided:
    - Cache hit: reloads code from registry, skipping generation entirely.
    - Cache miss: generates, verifies, then persists verified tools to registry.
    """
    specs = parse_new_tools(mediator_text)
    results = []

    # Extract MEDIATOR's overall rationale to give tool fixer more context
    _rationale = ""
    import re as _re, json as _json
    for _raw in _re.findall(r"```(?:json)?\s*(.*?)\s*```", mediator_text, _re.DOTALL | _re.IGNORECASE):
        try:
            _obj = _json.loads(_raw)
            if isinstance(_obj, dict) and "rationale" in _obj:
                _rationale = _obj["rationale"]
                break
        except Exception:
            pass

    for spec in specs:
        if _rationale and "rationale" not in spec:
            spec = dict(spec, rationale=_rationale)
        name = spec.get("name", "?")
        total_ms = 0
        final_success = False
        final_error = ""
        code = ""

        # ---- Registry cache check ----------------------------------------
        if tool_registry:
            cached = tool_registry.get(name)
            if cached:
                ok, err = register_dynamic_tool(name, cached["code"])
                log_fn(f"  [tool_creator] {name}: loaded from registry (task "
                       f"{cached.get('source_task', '?')})", force=True)
                results.append({
                    "name": name, "spec": spec, "code": cached["code"],
                    "success": ok, "error": err, "ms": 0,
                })
                if human_in_loop:
                    disp.show_tool_generation(spec, cached["code"], ok, err)
                continue

        # ---- Generate new tool -------------------------------------------
        log_fn(f"  [tool_creator] Generating tool: {name}...", force=True)
        code, gen_ms = await run_tool_generator(spec, task=task)
        total_ms = gen_ms
        fix_attempt = 0

        # Use args from spec so the tool is tested with the actual parameters
        # MEDIATOR intends to call it with (e.g. color_map, object_color, etc.)
        spec_args = spec.get("args", {})
        # Filter out non-serialisable / placeholder values; keep only concrete ones
        concrete_args = {k: v for k, v in spec_args.items()
                         if not isinstance(v, str) or not v.startswith("<")}

        if task:
            all_pass, trace = test_tool_code(name, code, task, default_args=concrete_args)
            accumulated_traces = [trace]  # grow across fix attempts for richer fixer context

            while not all_pass and fix_attempt < _TOOL_FIX_ATTEMPTS:
                fix_attempt += 1
                log_fn(f"  [tool_creator] {name}: demo verification failed, fixing "
                       f"(attempt {fix_attempt}/{_TOOL_FIX_ATTEMPTS})...", force=True)
                combined_trace = "\n\n---\n\n".join(
                    f"### Attempt {i+1} trace\n{t}" for i, t in enumerate(accumulated_traces)
                )
                fixed_code, fix_ms = await run_tool_generator_fix(spec, code, combined_trace, task=task)
                total_ms += fix_ms
                code = fixed_code
                all_pass, trace = test_tool_code(name, code, task, default_args=concrete_args)
                accumulated_traces.append(trace)

            if all_pass:
                final_success = True
                log_fn(f"  [tool_creator] {name}: verified OK "
                       f"({'first try' if fix_attempt == 0 else f'{fix_attempt} fix(es)'})",
                       force=True)
            else:
                final_success, final_error = register_dynamic_tool(name, code)
                log_fn(f"  [tool_creator] {name}: FAILED verification after "
                       f"{_TOOL_FIX_ATTEMPTS} fix(es) — registered with last attempt",
                       force=True)
                # Show a snippet of the last verification trace so operator can diagnose
                last_trace = accumulated_traces[-1] if accumulated_traces else ""
                if last_trace:
                    trace_lines = [l for l in last_trace.splitlines() if l.strip()]
                    snippet = "\n        ".join(trace_lines[:8])
                    log_fn(f"  [tool_creator] {name}: last trace snippet:\n        {snippet}",
                           force=True)
        else:
            final_success, final_error = register_dynamic_tool(name, code)
            log_fn(f"  [tool_creator] {name}: "
                   f"{'registered OK (no demo check)' if final_success else f'FAILED: {final_error}'}",
                   force=True)

        # ---- Persist to registry if verified ------------------------------
        if tool_registry and code:
            source_task = task.get("_task_id", "") if task else ""
            tool_registry.register(
                name=name,
                code=code,
                verified=final_success,
                source_task=source_task,
                description=spec.get("description", ""),
                fix_attempts=fix_attempt,
            )

        results.append({
            "name": name, "spec": spec, "code": code,
            "success": final_success, "error": final_error, "ms": total_ms,
        })
        if human_in_loop:
            disp.show_tool_generation(spec, code, final_success, final_error)

    return results


# ---------------------------------------------------------------------------
# Rule matching (Round 0)
# ---------------------------------------------------------------------------

TWO_STAGE_THRESHOLD = 30  # use two-stage retrieval when active task rules exceed this

async def match_rules(
    rule_engine: RuleEngine,
    task: dict,
) -> list[RuleMatch]:
    """Evaluate which rules match this puzzle. Returns ranked matches.

    When the active rule count exceeds TWO_STAGE_THRESHOLD, uses a two-stage
    approach to keep prompt size manageable:
      Stage 1 (cheap): ask the LLM which broad categories apply.
      Stage 2 (standard): run full matching only on the category-filtered subset.
    """
    active_task_rules = rule_engine.active_task_rules()
    if not active_task_rules:
        return []

    task_text = format_task_for_prompt(task)

    if len(active_task_rules) > TWO_STAGE_THRESHOLD:
        # --- Stage 1: category filter ---
        cat_prompt = rule_engine.build_category_filter_prompt(task_text)
        if cat_prompt:
            cat_text, _ms1 = await call_agent("MEDIATOR", cat_prompt, max_tokens=256)
            subset = rule_engine.filter_rules_by_categories(cat_text, max_rules=25)
        else:
            subset = active_task_rules[:25]   # fallback: first 25
        # --- Stage 2: full match on subset ---
        user_msg = rule_engine.build_match_prompt(task_text, rules_subset=subset)
    else:
        user_msg = rule_engine.build_match_prompt(task_text)

    text, _ms = await call_agent("MEDIATOR", user_msg, max_tokens=1024)
    return rule_engine.parse_match_response(text)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def run_ensemble(
    task: dict,
    task_id: str = "unknown",
    expected: Optional[Grid] = None,
    rule_engine: Optional[RuleEngine] = None,
    tool_registry: Optional[ToolRegistry] = None,
    human_in_loop: bool = False,
    human_hypothesis: str = "",
    human_insight: str = "",
    human_revision_hint: str = "",
    verbose: bool = True,
    dataset: str = "",
    solver_ids: list[str] | None = None,
    test_mode: bool = False,
) -> TaskMetadata:
    """
    Run the full ensemble on a single ARC-AGI task.

    Flow: Rule match -> Solvers (text) -> MEDIATOR (pseudo-code) -> EXECUTOR -> revise loop
    """
    if rule_engine is None:
        rule_engine = RuleEngine()
    if tool_registry is None:
        tool_registry = ToolRegistry()

    # Tag task dict with its ID so _generate_and_register_tools can record it
    task["_task_id"] = task_id

    reset_cost_tracker()
    _tools_generated: list[str] = []
    _human_hints_used = bool(human_hypothesis or human_insight or human_revision_hint)

    test_input = task["test"][0]["input"] if task.get("test") else []
    exp_shape = (len(expected), len(expected[0])) if expected and expected[0] else None

    meta = TaskMetadata(
        task_id=task_id,
        train_pairs=len(task.get("train", [])),
        test_shape=(len(test_input), len(test_input[0]) if test_input else 0),
        expected_shape=exp_shape,
    )

    def log(msg: str, force: bool = False) -> None:
        if verbose and (force or not human_in_loop):
            print(msg)

    log(f"\n{'-'*50}", force=True)
    log(f"Task: {task_id}  ({meta.train_pairs} demos, test {meta.test_shape[0]}x{meta.test_shape[1]})", force=True)
    log(f"Model: {DEFAULT_MODEL}  |  Rules: {rule_engine.stats_summary()}", force=True)

    # ------------------------------------------------------------------
    # Round 0 — Rule matching
    # ------------------------------------------------------------------
    log("Round 0: matching rules...", force=True)
    matched_rules = await match_rules(rule_engine, task)
    fired_ids = [m.rule_id for m in matched_rules]

    if matched_rules:
        log(f"  Matched {len(matched_rules)} rule(s): {fired_ids}", force=True)
    else:
        log("  No rules matched.", force=True)

    rules_prompt_section = rule_engine.format_fired_rules_for_prompt(matched_rules)
    preference_priors_section = rule_engine.format_preference_rules_for_solver()

    if human_in_loop:
        disp.show_rule_matches(matched_rules, rule_engine)

    # ------------------------------------------------------------------
    # Checkpoint 0 — Show puzzle, ask for human hypothesis
    # ------------------------------------------------------------------
    if human_in_loop:
        disp.show_puzzle(task, task_id, expected=expected)
        human_hypothesis = disp.human_hypothesis_checkpoint(task_id, prefill=human_hypothesis)
        if human_hypothesis:
            log(f"  Human hypothesis: {human_hypothesis[:80]}")

    # ------------------------------------------------------------------
    # Round 1 — Parallel solver hypotheses (TEXT ONLY)
    # ------------------------------------------------------------------
    log("Round 1: solvers proposing hypotheses...", force=True)
    t_r1 = time.time()
    r1_entries = await run_solvers_round1(
        task,
        prior_knowledge=rules_prompt_section,
        human_hypothesis=human_hypothesis,
        solver_ids=solver_ids,
        preference_priors=preference_priors_section,
    )
    meta.solvers_r1 = r1_entries
    log(f"  Done in {time.time()-t_r1:.1f}s", force=True)
    for e in r1_entries:
        log(f"  {e.agent}: {e.confidence}  rule={e.rule}", force=True)

    if human_in_loop:
        disp.show_r1_hypotheses(r1_entries)

    # ------------------------------------------------------------------
    # Round 2 — MEDIATOR synthesizes pseudo-code
    # ------------------------------------------------------------------
    human_r2_insight = human_insight  # use CLI --insight even without --human
    if human_in_loop:
        human_r2_insight = disp.human_post_hypotheses_checkpoint(prefill=human_insight)

    log("Round 2: MEDIATOR synthesizing pseudo-code...", force=True)
    t_r2 = time.time()

    rule_section = rule_engine.build_mediator_rule_section(matched_rules, success=True)
    tool_section = tool_registry.build_tool_section_for_prompt()

    mediator_text, pseudocode, mediator_ms = await run_mediator_synthesize(
        task=task,
        solver_entries=r1_entries,
        prior_knowledge=rules_prompt_section,
        human_insight=human_r2_insight or human_hypothesis,
        rule_section=rule_section,
        tool_section=tool_section,
    )
    log(f"  Done in {time.time()-t_r2:.1f}s  |  {len(pseudocode)} steps", force=True)
    for s in pseudocode:
        log(f"    Step {s.get('step', '?')}: {s.get('tool', '?')}({s.get('args', {})})")

    for r in await _generate_and_register_tools(
        mediator_text, human_in_loop, log, task=task, tool_registry=tool_registry
    ):
        if r["success"]:
            _tools_generated.append(r["name"])

    if human_in_loop:
        disp.show_pseudocode(pseudocode, mediator_text)

    # ------------------------------------------------------------------
    # Round 3 — EXECUTOR runs pseudo-code (deterministic)
    # ------------------------------------------------------------------
    log("Round 3: EXECUTOR running pseudo-code against demos...", force=True)

    exec_result: ExecutionResult = run_executor(pseudocode, task)
    attempt = 1
    failed_tool_names: list[str] = []  # accumulated across all revisions

    if exec_result.all_pass:
        log(f"  All demos passed.", force=True)
    else:
        demo_acc_parts = [f"demo{d.demo_index}={d.cell_acc*100:.0f}%" for d in exec_result.demos]
        log(f"  Initial execution failed ({', '.join(demo_acc_parts)})", force=True)

    if human_in_loop:
        disp.show_execution_result(exec_result, expected=expected)

    # ------------------------------------------------------------------
    # Revision loop — MEDIATOR revises on failure
    # ------------------------------------------------------------------
    while not exec_result.all_pass and attempt <= MAX_REVISIONS:
        # Per-demo failure detail
        log(f"  Revision {attempt}/{MAX_REVISIONS}:", force=True)
        for d in exec_result.demos:
            status = "PASS" if d.passed else f"FAIL ({d.cell_acc*100:.0f}% acc, {len(d.diff)} cells wrong)"
            log(f"    Demo {d.demo_index + 1}: {status}", force=True)
            if not d.passed:
                for sr in d.steps:
                    args_str = ", ".join(f"{k}={v!r}" for k, v in sr.args.items()) if sr.args else ""
                    step_status = "OK" if sr.success else f"ERROR: {sr.error}"
                    log(f"      Step {sr.step_num}: {sr.tool}({args_str}) -> {step_status}", force=True)
                if d.diff:
                    sample = ", ".join(f"({r},{c}) got {g} want {w}" for r, c, g, w in d.diff[:5])
                    log(f"      Diff sample: {sample}", force=True)

        trace_text = format_execution_trace(exec_result)

        human_revision_insight = human_revision_hint  # use CLI --revision-hint even without --human
        if human_in_loop:
            human_revision_insight = disp.human_revision_checkpoint(attempt, prefill=human_revision_hint)

        # Accumulate tool names from this failed pseudocode so MEDIATOR won't reuse them
        for s in pseudocode:
            tname = s.get("tool", "")
            if tname and tname not in failed_tool_names:
                failed_tool_names.append(tname)

        mediator_text, pseudocode, rev_ms = await run_mediator_revise(
            task=task,
            solver_entries=r1_entries,
            previous_pseudocode=pseudocode,
            execution_trace=trace_text,
            human_insight=human_revision_insight,
            failed_tools=failed_tool_names if attempt > 1 else None,
        )
        mediator_ms += rev_ms

        # Extract and log MEDIATOR rationale so operator can see why it chose this approach
        _med_rationale = ""
        import re as _re, json as _json
        for _raw in _re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", mediator_text or "", _re.DOTALL):
            try:
                _obj = _json.loads(_raw)
                if isinstance(_obj, dict) and "rationale" in _obj:
                    _med_rationale = _obj["rationale"]
                    break
            except Exception:
                pass
        if _med_rationale:
            log(f"  MEDIATOR rationale: {_med_rationale}", force=True)

        if pseudocode:
            log(f"  Revised: {len(pseudocode)} steps", force=True)
            for s in pseudocode:
                args_str = ", ".join(f"{k}={v!r}" for k, v in s.get("args", {}).items()) if s.get("args") else ""
                log(f"    Step {s.get('step','?')}: {s.get('tool','?')}({args_str})", force=True)
        else:
            log(f"  Revised: 0 steps (MEDIATOR produced no pseudocode)", force=True)

        for r in await _generate_and_register_tools(
            mediator_text, human_in_loop, log, task=task, tool_registry=tool_registry
        ):
            if r["success"]:
                _tools_generated.append(r["name"])
            elif not r.get("success") and r.get("error"):
                log(f"  [tool_creator] {r['name']}: final error — {r['error'][:200]}", force=True)

        if human_in_loop:
            disp.show_pseudocode(pseudocode, mediator_text, revision=attempt)

        exec_result = run_executor(pseudocode, task)
        attempt += 1

        if human_in_loop:
            disp.show_execution_result(exec_result, expected=expected)

    # ------------------------------------------------------------------
    # Build final metadata
    # ------------------------------------------------------------------
    final_answer = exec_result.test_output
    total_rounds = 2 + attempt  # R0 + R1 + R2 + executor attempts

    meta.mediator = MediatorDecision(
        round=total_rounds,
        answer=final_answer,
        rationale=mediator_text[:800] if mediator_text else "",
        converged=exec_result.all_pass,
        raw_response=mediator_text or "",
        duration_ms=mediator_ms,
    )
    meta.total_duration_ms = int(time.time() * 1000) - meta.start_ms
    meta.rounds_completed = total_rounds
    compute_outcome(meta, expected)

    # Leaderboard stats
    ct = get_cost_tracker()
    meta.model            = DEFAULT_MODEL
    meta.dataset          = dataset
    meta.human_hints_used = _human_hints_used
    meta.tools_generated  = list(dict.fromkeys(_tools_generated))  # deduplicated
    meta.matched_rule_ids = fired_ids
    meta.input_tokens          = ct.input_tokens
    meta.cache_creation_tokens = ct.cache_creation_tokens
    meta.cache_read_tokens     = ct.cache_read_tokens
    meta.output_tokens         = ct.output_tokens
    meta.api_calls             = ct.api_calls
    meta.cost_usd              = round(ct.cost_usd(), 6)

    success = meta.correct or False

    if test_mode:
        # In test mode: no learning, no persistence.
        # Rules and tools are read-only — existing knowledge is used but nothing new is saved.
        return meta

    # ------------------------------------------------------------------
    # Update rule stats
    # ------------------------------------------------------------------
    fired_ids = {m.rule_id for m in matched_rules}
    for m in matched_rules:
        if success:
            rule_engine.record_success(m.rule_id, task_id)
        else:
            rule_engine.record_failure(m.rule_id, task_id)

    # Increment tasks_seen for all active ns rules (staleness tracking)
    rule_engine.increment_tasks_seen(fired_ids=fired_ids)

    # ------------------------------------------------------------------
    # Parse MEDIATOR rule updates
    # ------------------------------------------------------------------
    rule_changes: list[dict] = []
    if mediator_text:
        rule_changes = rule_engine.parse_mediator_rule_updates(mediator_text, task_id)
        if rule_changes:
            log(f"  Rule updates: {len(rule_changes)} rule(s) created/modified", force=True)

    # ------------------------------------------------------------------
    # Preference rule extraction — fires when a human insight corrected
    # a wrong hypothesis and the corrected approach succeeded.
    #
    # This is a (wrong_hypothesis → correction → success) training event.
    # MEDIATOR distills it into a preference rule: a soft prior about
    # *which hypothesis property to prefer* when evidence is ambiguous.
    # The preference does not hard-code the answer — it biases the solver
    # toward human-natural reasoning properties (topology, perceptual
    # grouping, relative position) over computationally-easy but
    # non-human-natural ones (exact cell count, bounding box area).
    # ------------------------------------------------------------------
    if not success:
        # Persist the solver's natural (unguided) hypotheses so a future
        # correction run (--insight) can reference what was actually wrong.
        hyps = [e.rule[:400] for e in r1_entries if e.rule.strip()]
        if hyps:
            _save_failed_hypothesis(task_id, hyps)

    if success and human_insight:
        log("  Correction event detected: extracting preference rule...", force=True)
        # Prefer hypotheses from a prior FAILED run — they capture what the
        # solver naturally proposes without the corrective insight.
        prior_failed = _load_failed_hypotheses().get(task_id, [])
        wrong_hyps = prior_failed if prior_failed else [e.rule[:300] for e in r1_entries]
        correct_approach = (mediator_text or "")[:600]
        existing_prefs = rule_engine.format_preference_rules_for_solver()
        pref_text = await run_mediator_extract_preference(
            task_id=task_id,
            wrong_hypotheses=wrong_hyps,
            human_insight=human_insight,
            correct_approach=correct_approach,
            existing_preference_rules=existing_prefs,
        )
        pref_changes = rule_engine.parse_mediator_rule_updates(pref_text, task_id)
        if pref_changes:
            pref_ids = [r["id"] for r in pref_changes]
            log(f"  Preference rules created: {pref_ids}", force=True)
            rule_changes.extend(pref_changes)
        # Clean up — no need to keep the stale failed hypothesis
        _clear_failed_hypothesis(task_id)

    # ------------------------------------------------------------------
    # Candidate promotion — promote candidate rules that fired and succeeded
    # on this task (independent confirmation).
    # Only promote if this task is DIFFERENT from the rule's source_task.
    # ------------------------------------------------------------------
    if success and matched_rules:
        promoted = []
        for m in matched_rules:
            rule = rule_engine.get(m.rule_id)
            if rule and rule.get("status") == "candidate" and rule.get("source_task") != task_id:
                if rule_engine.promote_candidate(m.rule_id):
                    promoted.append(m.rule_id)
        if promoted:
            log(f"  Promoted candidate→active: {promoted}", force=True)

    # ------------------------------------------------------------------
    # Generalization pass — after a successful run with new task rules,
    # ask MEDIATOR to propose generalized candidate variants.
    # Only fires when task rules (not just preference rules) were created.
    # ------------------------------------------------------------------
    new_task_rules = [r for r in rule_changes if r.get("rule_type", "task") == "task"
                      and r.get("lineage", {}).get("type") == "new"]
    if success and new_task_rules:
        log("  Running generalization pass...", force=True)
        existing_summary = rule_engine.format_rules_for_matching()
        gen_text = await run_generalization_pass(
            task_id=task_id,
            new_rule_ids=[r["id"] for r in new_task_rules],
            new_rules=new_task_rules,
            existing_rules_summary=existing_summary,
        )
        gen_changes = rule_engine.parse_mediator_rule_updates(gen_text, task_id)
        if gen_changes:
            gen_ids = [r["id"] for r in gen_changes]
            log(f"  Generalized candidate rules created: {gen_ids}", force=True)
            rule_changes.extend(gen_changes)

    # Auto-deprecate/flag consistently failing or stale rules
    pruned = rule_engine.auto_deprecate()
    if pruned:
        log(f"  Auto-pruned {len(pruned)} rule(s): {pruned}", force=True)

    if verbose and not human_in_loop:
        print_task_summary(meta, expected)

    # Final display
    if human_in_loop:
        disp.show_final_result(meta, expected)
        if rule_changes or matched_rules:
            disp.show_rule_updates(matched_rules, rule_changes, success, rule_engine)

    return meta


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------

def run_ensemble_sync(
    task: dict,
    task_id: str = "unknown",
    expected: Optional[Grid] = None,
    rule_engine: Optional[RuleEngine] = None,
    human_in_loop: bool = False,
    verbose: bool = True,
) -> TaskMetadata:
    return asyncio.run(run_ensemble(
        task=task,
        task_id=task_id,
        expected=expected,
        rule_engine=rule_engine,
        human_in_loop=human_in_loop,
        verbose=verbose,
    ))
