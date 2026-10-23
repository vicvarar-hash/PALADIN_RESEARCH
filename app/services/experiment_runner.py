"""
Experiment runner — batch execution of experiment configurations.

Writes temporary policy files, builds a fresh DecisionEngine per config,
runs all (persona × task) evaluations, and computes aggregate metrics.
Supports both deterministic simulation and real LLM inference with smart
per-task caching (one API call per unique task, replayed across personas).
"""

import os
import json
import yaml
import hashlib
import shutil
import tempfile
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Callable, Any, Iterable
from collections import defaultdict

from app.services.experiment_config import (
    ExperimentConfig, EXPERIMENTS, EXPERIMENT_MAP,
    PERSONAS, LEGITIMATE_PAIRINGS, LEGITIMATE_PAIRINGS_NORMALIZED, simulate_llm_output,
)
from app.services.normalization import normalize_mcp_name
from app.services.spiffe_registry_service import SpiffeRegistryService
from app.services.spiffe_allowlist_service import SpiffeAllowlistService
from app.services.rbac_service import RBACService
from app.services.abac_rule_service import ABACRuleService
from app.services.tsphol_rule_service import TSPHOLRuleService
from app.services.mcp_attribute_service import MCPAttributeService
from app.services.decision_engine import DecisionEngine
from app.loaders.mcp_loader import load_mcp_personas


# ═══════════════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    experiment: str
    group: str
    persona: str
    task_idx: int
    domain: str
    match_tag: str
    is_legitimate: bool
    final_decision: str
    denial_source: Optional[str]
    identity_state: str
    transport_state: str
    rbac_state: str
    abac_state: str
    tsphol_state: str
    has_write: bool
    # LLM inference tracking
    inference_mode: str = "simulation"        # "simulation" or "llm"
    llm_failed: bool = False                  # True if LLM call errored
    llm_error: Optional[str] = None           # Error message if failed
    selected_tools: List[str] = field(default_factory=list)   # What was actually evaluated
    selected_mcps: List[str] = field(default_factory=list)    # What was actually evaluated
    groundtruth_tools: List[str] = field(default_factory=list)
    groundtruth_mcps: List[str] = field(default_factory=list)
    # LLM selection accuracy vs groundtruth
    tool_match: bool = False            # Exact set match (selected == groundtruth)
    tool_jaccard: float = 0.0           # Jaccard similarity (partial credit)
    # LLM validation verdict (validation mode only) — persisted so the
    # Post-Experiment Lab can replay the two issue-code-dependent TRAC rules
    # without re-invoking the LLM.
    issue_codes: List[str] = field(default_factory=list)
    is_valid: Optional[bool] = None


@dataclass
class ExperimentMetrics:
    name: str
    description: str
    total: int = 0
    allow_count: int = 0
    deny_count: int = 0
    deception_count: int = 0
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0
    identity_denials: int = 0
    transport_denials: int = 0
    rbac_denials: int = 0
    abac_denials: int = 0
    tsphol_denials: int = 0
    other_denials: int = 0
    llm_failures: int = 0
    # LLM selection accuracy
    tool_exact_matches: int = 0         # LLM picked exactly the groundtruth tools
    tool_jaccard_sum: float = 0.0       # Sum of Jaccard similarities (for averaging)
    tool_evaluated: int = 0             # Rows where LLM selection was compared

    @property
    def tool_accuracy(self) -> float:
        """Exact-match accuracy: fraction of tasks where LLM picked groundtruth tools."""
        return self.tool_exact_matches / self.tool_evaluated if self.tool_evaluated > 0 else 0.0

    @property
    def tool_jaccard_avg(self) -> float:
        """Average Jaccard similarity between selected and groundtruth tool sets."""
        return self.tool_jaccard_sum / self.tool_evaluated if self.tool_evaluated > 0 else 0.0

    @property
    def precision(self) -> float:
        d = self.true_positive + self.false_positive
        return self.true_positive / d if d > 0 else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positive + self.false_negative
        return self.true_positive / d if d > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def security_failure_rate(self) -> float:
        illegit = self.true_positive + self.false_negative
        return self.false_negative / illegit if illegit > 0 else 0.0

    @property
    def allow_rate(self) -> float:
        return self.allow_count / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(precision=self.precision, recall=self.recall, f1=self.f1,
                 security_failure_rate=self.security_failure_rate, allow_rate=self.allow_rate,
                 tool_accuracy=self.tool_accuracy, tool_jaccard_avg=self.tool_jaccard_avg)
        return d


# ═══════════════════════════════════════════════════════════════════════
# Engine builder with temp policy files
# ═══════════════════════════════════════════════════════════════════════

def _write_temp_policies(policies: Dict[str, dict], tmp_dir: str):
    """Write policy dicts to temp directory as files the services can load."""
    # Registry
    with open(os.path.join(tmp_dir, "spiffe_registry.json"), "w") as f:
        json.dump(policies["registry"], f, indent=2)
    # Allowlist
    with open(os.path.join(tmp_dir, "spiffe_allowlist.json"), "w") as f:
        json.dump(policies["allowlist"], f, indent=2)
    # RBAC
    with open(os.path.join(tmp_dir, "rbac.yaml"), "w") as f:
        yaml.dump(policies["rbac"], f, default_flow_style=False)
    # ABAC
    with open(os.path.join(tmp_dir, "abac_rules.yaml"), "w") as f:
        yaml.dump(policies["abac"], f, default_flow_style=False)
    # TRAC
    with open(os.path.join(tmp_dir, "trac_rules.yaml"), "w") as f:
        yaml.dump(policies["tsphol"], f, default_flow_style=False)


def build_engine_from_policies(policies: Dict[str, dict], personas_list) -> DecisionEngine:
    """Build a DecisionEngine from in-memory policy dicts using temp files."""
    tmp_dir = tempfile.mkdtemp(prefix="spiffi_exp_")
    try:
        _write_temp_policies(policies, tmp_dir)

        registry_svc = SpiffeRegistryService(filepath=os.path.join(tmp_dir, "spiffe_registry.json"))
        allowlist_svc = SpiffeAllowlistService(
            filepath=os.path.join(tmp_dir, "spiffe_allowlist.json"),
            registry_service=registry_svc,
        )
        rbac_svc = RBACService(filepath=os.path.join(tmp_dir, "rbac.yaml"))
        abac_rule_svc = ABACRuleService(filepath=os.path.join(tmp_dir, "abac_rules.yaml"))
        tsphol_svc = TSPHOLRuleService(filepath=os.path.join(tmp_dir, "trac_rules.yaml"))
        # MCPAttributeService reads from policy_dir, copy original attributes file
        orig_attrs = "policies/mcp_attributes.yaml"
        if os.path.exists(orig_attrs):
            shutil.copy2(orig_attrs, os.path.join(tmp_dir, "mcp_attributes.yaml"))
        # Also copy heuristic_policy.json and inference_config.json
        for fname in ["heuristic_policy.json", "inference_config.json",
                       "domain_capability_catalog.json", "required_capability_rules.json",
                       "mcp_risk_levels.yaml"]:
            src = os.path.join("policies", fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp_dir, fname))
        attribute_svc = MCPAttributeService(policy_dir=tmp_dir)

        engine = DecisionEngine(
            registry_svc=registry_svc,
            allowlist_svc=allowlist_svc,
            rbac_svc=rbac_svc,
            tsphol_svc=tsphol_svc,
            attribute_svc=attribute_svc,
            personas=personas_list,
            abac_rule_svc=abac_rule_svc,
        )
        # Keep reference to tmp_dir for cleanup
        engine._tmp_dir = tmp_dir
        return engine
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def cleanup_engine(engine: DecisionEngine):
    """Remove temp policy files after experiment completes."""
    tmp_dir = getattr(engine, "_tmp_dir", None)
    if tmp_dir and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
# Single evaluation
# ═══════════════════════════════════════════════════════════════════════

WRITE_KEYWORDS = {"create", "update", "delete", "transition", "place", "add", "set", "remove"}


def run_single(engine: DecisionEngine, persona_key: str, task,
               task_idx: int, config: ExperimentConfig, mode: str = "selection",
               llm_output: Optional[dict] = None) -> RunResult:
    """Run one (persona, task) pair through the decision engine.
    
    Accepts either a raw dict (from JSON) or an AstraTask pydantic model.
    If llm_output is provided, uses its selected tools/mcps for governance
    evaluation (real LLM mode). Otherwise falls back to simulation.
    """
    persona = PERSONAS[persona_key]
    spiffe_id = persona["spiffe_id"]

    # Extract groundtruth / task metadata (never changes)
    if isinstance(task, dict):
        gt_tools = task["input"]["tools"]
        gt_mcps = task["input"]["mcp_servers"]
        task_text = task["input"]["task"]
        match_tag = task.get("match_tag", "null")
    else:
        gt_tools = task.candidate_tools
        gt_mcps = task.candidate_mcp
        task_text = task.task
        match_tag = getattr(task, "match_tag", "null")

    task_domain = normalize_mcp_name(gt_mcps[0]) if gt_mcps else "unknown"
    domain_authorized = task_domain in LEGITIMATE_PAIRINGS_NORMALIZED.get(persona_key, set())
    is_legitimate = domain_authorized and match_tag == "correct"

    # Determine inference source and the bundle to evaluate
    inference_mode = "simulation"
    llm_failed = False
    llm_error = None

    if llm_output is not None:
        inference_mode = "llm"
        if llm_output.get("_failed"):
            # LLM call errored — mark as failed, skip governance
            llm_failed = True
            llm_error = llm_output.get("_error", "Unknown LLM error")
            return RunResult(
                experiment=config.name, group=config.group,
                persona=persona_key, task_idx=task_idx,
                domain=task_domain, match_tag=match_tag,
                is_legitimate=is_legitimate,
                final_decision="LLM_FAILED", denial_source=None,
                identity_state="N/A", transport_state="N/A",
                rbac_state="N/A", abac_state="N/A", tsphol_state="N/A",
                has_write=False,
                inference_mode="llm", llm_failed=True, llm_error=llm_error,
                selected_tools=[], selected_mcps=[],
                groundtruth_tools=gt_tools, groundtruth_mcps=gt_mcps,
            )
        # Two cache shapes are supported:
        #   • Selection cache  → LLM picks tools, governance evaluates picks.
        #   • Validation cache → LLM judges the candidate (GT) bundle,
        #     governance evaluates the GT bundle itself (matches Prediction Lab).
        cache_mode = llm_output.get("_mode", "selection")
        if cache_mode == "validation":
            eval_tools = gt_tools
            eval_mcps = gt_mcps
        else:
            eval_tools = llm_output.get("selected_tools", gt_tools)
            eval_mcps = llm_output.get("selected_mcps", gt_mcps)
        llm_out = llm_output
    else:
        # Simulation: use groundtruth tools (deterministic passthrough)
        llm_out = simulate_llm_output(task, mode=mode, seed_extra=persona_key)
        eval_tools = llm_out.get("selected_tools", gt_tools)
        eval_mcps = llm_out.get("selected_mcps", gt_mcps)

    mcp_filter = gt_mcps[0] if gt_mcps else "All"

    pre_llm = engine.pre_llm_check(spiffe_id, eval_mcps, eval_tools)

    result = engine.evaluate(
        pre_llm_result=pre_llm,
        caller_spiffe_id=spiffe_id,
        mcps=eval_mcps,
        tools=eval_tools,
        llm_outputs=llm_out,
        task_text=task_text,
        mode=mode,
        mcp_filter=mcp_filter,
    )

    has_write = any(kw in t for t in eval_tools for kw in WRITE_KEYWORDS)

    # LLM selection accuracy vs groundtruth
    sel_set = set(eval_tools)
    gt_set = set(gt_tools)
    tool_match = sel_set == gt_set
    union = sel_set | gt_set
    tool_jaccard = len(sel_set & gt_set) / len(union) if union else 1.0

    # Persist the LLM validation verdict (validation mode) for faithful replay.
    issue_codes = list(llm_out.get("issue_codes", []) or []) if isinstance(llm_out, dict) else []
    is_valid = llm_out.get("is_valid") if isinstance(llm_out, dict) else None

    return RunResult(
        experiment=config.name,
        group=config.group,
        persona=persona_key,
        task_idx=task_idx,
        domain=task_domain,
        match_tag=match_tag,
        is_legitimate=is_legitimate,
        final_decision=result.final_decision,
        denial_source=result.denial_source,
        identity_state=result.evaluation_states.get("identity", "N/A"),
        transport_state=result.evaluation_states.get("transport", "N/A"),
        rbac_state=result.evaluation_states.get("rbac", "N/A"),
        abac_state=result.evaluation_states.get("abac", "N/A"),
        tsphol_state=result.evaluation_states.get("tsphol", "N/A"),
        has_write=has_write,
        inference_mode=inference_mode,
        llm_failed=False,
        llm_error=None,
        selected_tools=list(eval_tools),
        selected_mcps=list(eval_mcps),
        groundtruth_tools=gt_tools,
        groundtruth_mcps=gt_mcps,
        tool_match=tool_match,
        tool_jaccard=tool_jaccard,
        issue_codes=issue_codes,
        is_valid=is_valid,
    )


# ═══════════════════════════════════════════════════════════════════════
# Metrics computation
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(results: List[RunResult], config: ExperimentConfig) -> ExperimentMetrics:
    m = ExperimentMetrics(name=config.name, description=config.description)
    m.total = len(results)

    for r in results:
        # Skip LLM failures from governance metrics
        if r.llm_failed:
            m.llm_failures += 1
            continue

        # LLM selection accuracy (computed for every non-failed row)
        m.tool_evaluated += 1
        if r.tool_match:
            m.tool_exact_matches += 1
        m.tool_jaccard_sum += r.tool_jaccard

        is_denied = r.final_decision in ("DENY", "DECEPTION_ROUTED")
        is_allowed = not is_denied

        if r.final_decision == "DECEPTION_ROUTED":
            m.deception_count += 1
            m.deny_count += 1
        elif is_denied:
            m.deny_count += 1
        else:
            m.allow_count += 1

        if r.is_legitimate:
            if is_allowed:
                m.true_negative += 1
            else:
                m.false_positive += 1
        else:
            if is_denied:
                m.true_positive += 1
            else:
                m.false_negative += 1

        if is_denied and r.denial_source:
            src = r.denial_source.lower()
            if "identity" in src:
                m.identity_denials += 1
            elif "transport" in src:
                m.transport_denials += 1
            elif "rbac" in src:
                m.rbac_denials += 1
            elif "abac" in src:
                m.abac_denials += 1
            elif any(k in src for k in ("ts-phol", "tsphol", "ts_phol")):
                m.tsphol_denials += 1
            else:
                m.other_denials += 1

    return m


def compute_domain_breakdown(results: List[RunResult]) -> Dict[str, Dict[str, int]]:
    """Per-domain metrics breakdown."""
    by_domain: Dict[str, list] = defaultdict(list)
    for r in results:
        by_domain[r.domain].append(r)

    breakdown = {}
    for domain, domain_results in sorted(by_domain.items()):
        tp = sum(1 for r in domain_results if not r.is_legitimate and r.final_decision in ("DENY", "DECEPTION_ROUTED"))
        tn = sum(1 for r in domain_results if r.is_legitimate and r.final_decision == "ALLOW")
        fp = sum(1 for r in domain_results if r.is_legitimate and r.final_decision in ("DENY", "DECEPTION_ROUTED"))
        fn = sum(1 for r in domain_results if not r.is_legitimate and r.final_decision == "ALLOW")
        total = len(domain_results)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        breakdown[domain] = {"total": total, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                              "precision": p, "recall": rec, "f1": f1}
    return breakdown


# ═══════════════════════════════════════════════════════════════════════
# LLM inference cache — one API call per unique task
# ═══════════════════════════════════════════════════════════════════════

def _task_fingerprint(task) -> str:
    """Stable fingerprint for a task — used as cache key.

    Includes ``match_tag`` so rows with identical (text, mcps) but
    different bundle category (correct vs wrong vs null) map to distinct
    fingerprints. This keeps train/test/other partitions disjoint and
    prevents cache collisions between correct and adversarial rows.
    """
    if isinstance(task, dict):
        text = task["input"]["task"]
        mcps = task["input"]["mcp_servers"]
        tag = task.get("match_tag") or "null"
    else:
        text = task.task
        mcps = task.candidate_mcp
        tag = getattr(task, "match_tag", None) or "null"
    raw = f"{text}|{','.join(sorted(mcps))}|{tag}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _to_astra_task(task):
    """Convert raw dict to AstraTask if needed."""
    if not isinstance(task, dict):
        return task
    from app.models.astra import AstraTask
    inp = task["input"]
    return AstraTask(
        task=inp["task"],
        candidate_tools=inp["tools"],
        candidate_mcp=inp["mcp_servers"],
        groundtruth_tools=task.get("expected_output", {}).get("tools", inp["tools"]),
        groundtruth_mcp=task.get("expected_output", {}).get("mcp_servers", inp["mcp_servers"]),
        match_tag=task.get("match_tag", "null"),
    )


def build_llm_cache(tasks: list, personas_list, api_key: str,
                    model: str = "gpt-4o",
                    provider: str = "openai",
                    mode: str = "selection",
                    progress_callback: Optional[Callable] = None,
                    max_retries: int = 2,
                    retriever: Optional[Any] = None,
                    rbac_scoped: bool = False,
                    capability_filter: bool = False, *,
                    endpoint: Optional[str] = None,
                    api_version: Optional[str] = None) -> Dict[str, dict]:
    """Call the LLM once per unique task and cache results.

    Behavior depends on ``mode``:

    * ``"selection"`` — invokes :class:`PredictionService` to have the LLM
      pick a 3-tool bundle. Governance later evaluates the LLM's picks.
    * ``"validation"`` — invokes :class:`ValidationService` to have the LLM
      judge the dataset's candidate (groundtruth) bundle. Governance later
      evaluates the candidate bundle itself (mirrors Prediction Lab).

    If ``retriever`` (an :class:`ExemplarRetriever`) is provided, K RA-ICL
    exemplars are injected into each prompt.

    Prompt-context optimizations (selection mode only):

    * ``rbac_scoped`` — per-persona RBAC scoping: hide MCPs the persona is
      not allowed to use. This expands the cache by ~5x (one entry per
      distinct allowed-MCP set across the 6 personas). Cache keys become
      ``f"{fp}|{mcps_key}"`` where mcps_key is the sorted allowed-MCP set.
    * ``capability_filter`` — drop tools whose classified capabilities don't
      intersect the task's required/optional capability set. Does not affect
      cache cardinality (task-deterministic).

    Returns dict mapping cache-key → llm_output dict compatible with
    run_single()'s llm_output parameter. The matching key-builder is exposed
    as :func:`llm_cache_key`.
    """
    from app.services.llm_provider import LLMProvider
    from app.services.prediction_service import PredictionService
    from app.services.validation_service import ValidationService
    from app.services.intent_engine import IntentEngine
    from app.services.experiment_config import LEGITIMATE_PAIRINGS

    llm = LLMProvider(api_key=api_key, model=model, provider=provider,
                      endpoint=endpoint, api_version=api_version)
    if not llm.is_configured():
        raise ValueError(llm.configuration_error)

    if mode not in ("selection", "validation"):
        raise ValueError(f"Unsupported mode: {mode!r} (expected 'selection' or 'validation')")

    # Validation mode judges the candidate bundle and is persona-agnostic;
    # prompt-context flags only apply to selection.
    if mode == "validation":
        rbac_scoped = False
        capability_filter = False

    intent_engine = IntentEngine()
    pred_svc = PredictionService(llm=llm, personas=personas_list,
                                  intent_engine=intent_engine)
    val_svc = ValidationService(llm=llm, personas=personas_list)

    # Dedupe (task, allowed_mcps_set) pairs. When rbac_scoped is OFF, the
    # set is empty → one entry per unique task (legacy behavior).
    unique_jobs: Dict[str, tuple] = {}
    if rbac_scoped:
        # Collect distinct allowed-MCP sets across the active personas.
        from app.services.experiment_config import PERSONAS
        distinct_mcp_sets = set()
        for pkey in PERSONAS.keys():
            allowed = frozenset(LEGITIMATE_PAIRINGS.get(pkey, set()))
            if allowed:
                distinct_mcp_sets.add(allowed)
        for task in tasks:
            fp = _task_fingerprint(task)
            for allowed in distinct_mcp_sets:
                key = llm_cache_key(fp, allowed)
                if key not in unique_jobs:
                    unique_jobs[key] = (task, allowed)
    else:
        for task in tasks:
            fp = _task_fingerprint(task)
            key = llm_cache_key(fp, None)
            if key not in unique_jobs:
                unique_jobs[key] = (task, None)

    cache: Dict[str, dict] = {}
    total = len(unique_jobs)
    done = 0
    errors = 0

    for key, (task, allowed) in unique_jobs.items():
        astra_task = _to_astra_task(task)
        exemplars = retriever.get(task) if retriever is not None else None
        raicl_k = len(exemplars) if exemplars else 0

        for attempt in range(max_retries + 1):
            try:
                if mode == "selection":
                    sel = pred_svc.run_selection(
                        astra_task,
                        exemplars=exemplars,
                        allowed_mcps=set(allowed) if allowed else None,
                        capability_filter=capability_filter,
                    )
                    if sel.validation_errors and "LLM_NOT_CONFIGURED" in sel.validation_errors:
                        cache[key] = {"_failed": True, "_error": "LLM not configured"}
                        break

                    # Surface silent in-method failures: if run_selection caught
                    # an exception internally it returns empty tools with the
                    # error tagged into justification. Promote that to a real
                    # failure so llm_failed/llm_error reflect it in row logs.
                    silent_err = None
                    if (not sel.selected_tools) and isinstance(sel.justification, str) \
                            and sel.justification.startswith("Error during selection:"):
                        silent_err = sel.justification.replace("Error during selection:", "").strip()
                    if silent_err is None and sel.validation_errors:
                        # Heuristic: validation_errors that look like exception strings
                        for ve in sel.validation_errors:
                            if isinstance(ve, str) and (":" in ve and ve.split(":", 1)[0].endswith("Error")):
                                silent_err = ve
                                break
                    if silent_err:
                        cache[key] = {"_failed": True, "_error": silent_err,
                                      "justification": sel.justification,
                                      "validation_errors": sel.validation_errors}
                        errors += 1
                        break

                    cache[key] = {
                        "_mode": "selection",
                        "_raicl_k": raicl_k,
                        "_rbac_scoped": bool(allowed),
                        "_cap_filtered": bool(capability_filter),
                        "_allowed_mcps": sorted(allowed) if allowed else None,
                        "selected_tools": sel.selected_tools,
                        "selected_mcps": sel.selected_mcp,
                        "justification": sel.justification,
                        "id_source": "LLM",
                        "expected_domain": normalize_mcp_name(
                            astra_task.candidate_mcp[0]) if astra_task.candidate_mcp else "uncertain",
                        "raw_output": sel.raw_output,
                        "validation_errors": sel.validation_errors,
                    }
                else:  # validation
                    val = val_svc.run_validation(astra_task, exemplars=exemplars)
                    if val.issues and "LLM_NOT_CONFIGURED" in val.issues:
                        cache[key] = {"_failed": True, "_error": "LLM not configured"}
                        break

                    cache[key] = {
                        "_mode": "validation",
                        "_raicl_k": raicl_k,
                        "selected_tools": list(astra_task.candidate_tools),
                        "selected_mcps": list(astra_task.candidate_mcp),
                        "justification": val.reason,
                        "is_valid": val.is_valid,
                        "issue_codes": val.issue_codes,
                        "issues": val.issues,
                        "expected_domain": val.expected_domain,
                        "actual_domain": val.actual_domain,
                        "task_alignment_score": val.task_alignment_score,
                        "id_source": "LLM",
                        "raw_output": val.raw_output,
                    }
                break
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))  # backoff
                    continue
                cache[key] = {"_failed": True, "_error": str(e)}
                errors += 1
                break

        done += 1
        if progress_callback:
            progress_callback({
                "phase": "llm_cache",
                "current": done,
                "total": total,
                "errors": errors,
            })

    return cache


def llm_cache_key(fp: str, allowed_mcps: Optional[Iterable[str]]) -> str:
    """Build the cache key used by build_llm_cache and run_experiment lookup.

    When ``allowed_mcps`` is falsy, returns the plain fingerprint (legacy).
    When provided, appends a stable suffix derived from the sorted MCP set,
    so personas that share the same allow-list share the cached LLM result.
    """
    if not allowed_mcps:
        return fp
    suffix = "+".join(sorted(set(allowed_mcps)))
    return f"{fp}|{suffix}"


# ═══════════════════════════════════════════════════════════════════════
# Batch experiment runner
# ═══════════════════════════════════════════════════════════════════════

def run_experiment(config: ExperimentConfig, tasks: list, personas_list,
                   mode: str = "selection",
                   progress_callback: Optional[Callable] = None,
                   llm_cache: Optional[Dict[str, dict]] = None,
                   task_filter_fingerprints: Optional[set] = None,
                   rbac_scoped_cache: bool = False) -> tuple:
    """
    Run a complete experiment: build engine, iterate all persona×task pairs, compute metrics.

    If llm_cache is provided, uses real LLM results for governance evaluation.
    Otherwise falls back to deterministic simulation.

    If ``task_filter_fingerprints`` is provided, only tasks whose fingerprint
    is in that set are evaluated (used to restrict governance evaluation to
    the held-out test split when RA-ICL is enabled).

    If ``rbac_scoped_cache`` is True, the LLM cache is assumed to be keyed by
    ``(task_fingerprint, allowed_mcps_set)`` and is looked up using the
    persona's allow-list from LEGITIMATE_PAIRINGS.

    Returns: (metrics: ExperimentMetrics, results: List[RunResult])
    """
    from app.services.experiment_config import LEGITIMATE_PAIRINGS
    policies = config.get_policies()
    engine = build_engine_from_policies(policies, personas_list)

    try:
        results: List[RunResult] = []
        active_personas = list(PERSONAS.keys())

        # Apply match_tag filter if configured
        if config.match_tag_filter:
            filtered_tasks = []
            for t in tasks:
                tag = t.get("match_tag", "null") if isinstance(t, dict) else getattr(t, "match_tag", "null")
                if tag == config.match_tag_filter:
                    filtered_tasks.append(t)
        else:
            filtered_tasks = tasks

        # Apply test-split fingerprint filter on top of match_tag filter
        if task_filter_fingerprints is not None:
            filtered_tasks = [t for t in filtered_tasks
                              if _task_fingerprint(t) in task_filter_fingerprints]

        total_evals = len(active_personas) * len(filtered_tasks)
        done = 0

        for persona_key in active_personas:
            persona_allowed = frozenset(LEGITIMATE_PAIRINGS.get(persona_key, set())) if rbac_scoped_cache else None
            for task_idx, task in enumerate(filtered_tasks):
                # Look up cached LLM output if available
                llm_out = None
                if llm_cache is not None:
                    fp = _task_fingerprint(task)
                    key = llm_cache_key(fp, persona_allowed)
                    llm_out = llm_cache.get(key)
                    # Defensive fallback: if scoped key missed, try plain fp
                    # (covers legacy caches loaded without rbac_scoped flag).
                    if llm_out is None and rbac_scoped_cache:
                        llm_out = llm_cache.get(fp)

                result = run_single(engine, persona_key, task, task_idx, config,
                                    mode=mode, llm_output=llm_out)
                results.append(result)
                done += 1
                if progress_callback and done % 50 == 0:
                    progress_callback({
                        "phase": "governance",
                        "current": done,
                        "total": total_evals,
                        "config": config.name,
                    })

        if progress_callback:
            progress_callback({
                "phase": "governance",
                "current": total_evals,
                "total": total_evals,
                "config": config.name,
            })

        metrics = compute_metrics(results, config)
        return metrics, results
    finally:
        cleanup_engine(engine)
