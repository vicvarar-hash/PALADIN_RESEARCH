"""LLM-inference producer for the Experiment LLM Lab.

Runs the LLM proposer ONCE per task (selection or validation) over the ASTRA
dataset and emits an ``llm_inference_v1`` log — no governance, no E1–E4. The
Post-Experiment Lab (the deterministic disposer) re-derives every RBAC/ABAC/TRAC
decision from these bundles.

Optionally augments each prompt with BM25 retrieval-augmented exemplars over a
70/30 train/test split (K=25 by default), in which case only the held-out test
cohort (plus all wrong/null tasks) is inferred — exactly the cohort the lab scores.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional

from app.services.experiment_runner import (
    build_llm_cache, llm_cache_key, _task_fingerprint,
)
from app.services.llm_inference_log import build_log, save_log, retrieval_meta, SCHEMA
from app.services.normalization import normalize_mcp_name  # noqa: F401  (kept for parity callers)

DEFAULT_BM25_K = 25
INFERENCE_DIR = os.path.join("datasets", "llm_inference_logs")


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s or "")).strip("-").lower()


def _match_tag(task) -> str:
    if isinstance(task, dict):
        return task.get("match_tag", "null")
    return getattr(task, "match_tag", "null")


def _candidate_tools(task) -> List[str]:
    if isinstance(task, dict):
        return list(task.get("input", {}).get("tools", []) or [])
    return list(getattr(task, "candidate_tools", []) or [])


def produce_inference_log(
    tasks: list,
    personas,
    *,
    model: str,
    provider: str,
    api_key: str,
    mode: str,
    use_bm25_raicl: bool = False,
    k: int = DEFAULT_BM25_K,
    progress_cb: Optional[Callable[[dict], None]] = None,
    endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
) -> dict:
    """Run LLM inference over the dataset and return an ``llm_inference_v1`` log dict.

    ``mode`` is ``"selection"`` (LLM picks the bundle) or ``"validation"`` (LLM judges
    the candidate bundle). When ``use_bm25_raicl`` is set, BM25 exemplars from the 70%
    train split are injected and only the 30% test cohort (+ wrong/null) is inferred.
    """
    if mode not in ("selection", "validation"):
        raise ValueError(f"mode must be 'selection' or 'validation', got {mode!r}")

    retriever = None
    test_fps = None
    retrieval = {"strategy": "none"}
    if use_bm25_raicl:
        from app.services.split_service import load_or_build_split
        from app.services.exemplar_retriever import ExemplarRetriever

        split_info = load_or_build_split(tasks, ratio=0.7, seed=42)
        train_pool = split_info.filter_train(tasks)
        retriever = ExemplarRetriever(
            train_pool=train_pool, k=k, strategy="bm25", pad_cross_domain=True,
        )
        test_fps = set(split_info.test_fingerprints) | set(split_info.other_fingerprints)
        retrieval = retrieval_meta("bm25", k, f"{split_info.ratio}@seed{split_info.seed}")

    infer_tasks = [t for t in tasks if (test_fps is None or _task_fingerprint(t) in test_fps)]

    cache = build_llm_cache(
        infer_tasks, personas, api_key=api_key, model=model, provider=provider,
        mode=mode, progress_callback=progress_cb, retriever=retriever,
        endpoint=endpoint, api_version=api_version,
    )

    out_tasks: List[dict] = []
    for ti, task in enumerate(tasks):
        fp = _task_fingerprint(task)
        if test_fps is not None and fp not in test_fps:
            continue
        entry = cache.get(llm_cache_key(fp, None))
        if entry is None:
            continue
        mtag = _match_tag(task)
        if entry.get("_failed"):
            out_tasks.append({
                "task_idx": ti, "selected_tools": [], "selected_mcps": [],
                "is_valid": None, "issue_codes": None, "match_tag": mtag,
                "tool_match": None, "tool_jaccard": None, "llm_failed": True,
            })
            continue
        sel_tools = list(entry.get("selected_tools") or [])
        sel_mcps = list(entry.get("selected_mcps") or [])
        # Selection accuracy vs the ASTRA candidate bundle (identical to run_single).
        sel_set, gt_set = set(sel_tools), set(_candidate_tools(task))
        union = sel_set | gt_set
        out_tasks.append({
            "task_idx": ti,
            "selected_tools": sel_tools,
            "selected_mcps": sel_mcps,
            "is_valid": entry.get("is_valid"),
            "issue_codes": entry.get("issue_codes"),
            "match_tag": mtag,
            "tool_match": (sel_set == gt_set),
            "tool_jaccard": (len(sel_set & gt_set) / len(union) if union else 1.0),
            "llm_failed": False,
        })

    return build_log(model=model, mode=mode, tasks=out_tasks, provider=provider,
                     retrieval=retrieval)


def default_log_path(log: dict) -> str:
    """A clean, descriptive path under datasets/llm_inference_logs/ for a produced log."""
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    ra = log.get("retrieval") or {}
    ra_tag = ""
    if ra.get("strategy") not in (None, "none", ""):
        ra_tag = f"_ra-{_slug(ra.get('strategy'))}" + (f"-k{ra.get('k')}" if ra.get("k") else "")
    name = f"{stamp}_{_slug(log.get('model'))}_{_slug(log.get('mode'))}{ra_tag}.json"
    return os.path.join(INFERENCE_DIR, name)


def produce_and_save(tasks, personas, **kw) -> tuple:
    """Convenience: produce a log and persist it. Returns (path, log)."""
    log = produce_inference_log(tasks, personas, **kw)
    path = save_log(default_log_path(log), log)
    return path, log
