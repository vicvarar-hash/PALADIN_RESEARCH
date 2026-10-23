"""Experiment LLM Lab — collect LLM inferences for the Post-Experiment Lab.

Runs the LLM *proposer* once per task over the ASTRA dataset and saves an
``llm_inference_v1`` log. Two modes:

* **Selection** — the LLM picks a tool bundle for the task.
* **Validation** — the LLM rules the dataset's candidate bundle valid / invalid.

No governance is computed here. The saved log feeds the **Post-Experiment Lab**
(the deterministic *disposer*), which re-derives the full RBAC · ABAC · TRAC
stack from the recorded bundles and lets you edit policies and compare.

Optionally augments each prompt with BM25 retrieval-augmented exemplars
(K=25) over a 70/30 train/test split, in which case only the held-out test
cohort (plus all wrong/null tasks) is inferred.
"""

from collections import Counter

import streamlit as st
import pandas as pd

from app.services.llm_inference_producer import produce_and_save, DEFAULT_BM25_K


def render_experiment_lab(tasks, personas):
    st.title("🧪 Experiment LLM Lab")
    st.markdown(
        "Collect **LLM inferences** for every task in the dataset — *selection* (the LLM "
        "picks the tool bundle) or *validation* (the LLM judges the candidate bundle). "
        "No governance runs here: the saved `llm_inference_v1` log feeds the "
        "**📊 Post-Experiment Lab**, which re-derives the full RBAC · ABAC · TRAC stack "
        "from these bundles and lets you edit policies and compare baseline vs modified."
    )

    # Provider / model / key come from the sidebar (shared with the Prediction Lab).
    from app.ui.llm_settings import get_llm_config, get_llm_config_error
    cfg = get_llm_config()
    provider, model, api_key = cfg["provider"], cfg["model"], cfg["api_key"]
    configuration_error = get_llm_config_error(cfg)
    if configuration_error:
        st.info(configuration_error)
    else:
        st.success(f"Using **{provider}** / `{model}` (configured in the sidebar)")

    c1, c2 = st.columns([2, 3])
    with c1:
        mode_label = st.radio(
            "Inference mode", ["Selection", "Validation"], horizontal=True,
            help="**Selection** — the LLM picks a tool bundle for each task. "
                 "**Validation** — the LLM rules the ASTRA candidate bundle valid/invalid.",
        )
        mode = "selection" if mode_label == "Selection" else "validation"
    with c2:
        use_raicl = st.checkbox(
            f"📚 BM25 retrieval-augmented exemplars (K={DEFAULT_BM25_K}, 70/30 split)",
            value=False,
            help="Inject K=25 BM25-ranked exemplars from the 70% train split into each "
                 "prompt, and infer only the 30% held-out test cohort (plus all wrong/null "
                 "tasks). When off, the full dataset is inferred with no exemplars.",
        )

    cohort = "30% test cohort (+ wrong/null)" if use_raicl else f"full dataset · {len(tasks):,} tasks"
    st.caption(f"One LLM call per task · {cohort}.")

    run = st.button("🤖 Run LLM inference", type="primary", disabled=bool(configuration_error),
                    use_container_width=True)

    if run:
        prog = st.progress(0.0, text="Calling the LLM…")

        def cb(info):
            cur, tot = info.get("current", 0), info.get("total", 1)
            errs = info.get("errors", 0)
            prog.progress(
                min(cur / tot, 1.0) if tot else 0.0,
                text=f"LLM inference — {cur}/{tot}" + (f" · ⚠️ {errs} errors" if errs else ""),
            )

        try:
            path, log = produce_and_save(
                tasks, personas, model=model, provider=provider, api_key=api_key,
                endpoint=cfg["endpoint"], api_version=cfg["api_version"],
                mode=mode, use_bm25_raicl=use_raicl, k=DEFAULT_BM25_K, progress_cb=cb,
            )
        except Exception as e:  # noqa: BLE001 — surface any provider/setup error to the UI
            prog.empty()
            st.error(f"❌ LLM inference failed: {e}")
            return
        prog.empty()
        st.session_state["llm_lab_result"] = {"path": path, "log": log}

    if "llm_lab_result" in st.session_state:
        _display(st.session_state["llm_lab_result"])
    else:
        st.info("Pick a mode and click **Run LLM inference** to begin.")


def _display(res: dict):
    log, path = res["log"], res["path"]
    rows = log["tasks"]
    st.success(f"📄 Saved `{path}` — **{len(rows)}** tasks · {log['mode']} · {log['model']}")

    ok = [t for t in rows if not t.get("llm_failed")]
    failed = len(rows) - len(ok)

    cols = st.columns(4)
    cols[0].metric("Tasks inferred", len(rows))
    cols[1].metric("LLM failures", failed)
    if log["mode"] == "validation":
        valid = sum(1 for t in ok if t.get("is_valid"))
        cols[2].metric("LLM “valid”", valid)
        cols[3].metric("LLM “invalid”", len(ok) - valid)
    else:
        tm = sum(1 for t in ok if t.get("tool_match"))
        jac = [t.get("tool_jaccard") or 0.0 for t in ok]
        cols[2].metric("Exact tool-match", tm)
        cols[3].metric("Avg tool Jaccard", f"{(sum(jac) / len(jac) if jac else 0.0):.2f}")

    tagc = Counter(t.get("match_tag") or "null" for t in rows)
    st.markdown("**Tasks by `match_tag`**")
    st.dataframe(
        pd.DataFrame(sorted(tagc.items()), columns=["match_tag", "tasks"]),
        hide_index=True, use_container_width=True,
    )

    ra = log.get("retrieval") or {}
    if ra.get("strategy") not in (None, "none", ""):
        st.caption(f"Retrieval: **{ra.get('strategy')}** · K={ra.get('k')} · split=`{ra.get('split')}`")
    st.info(
        "Open the **📊 Post-Experiment Lab** to govern these bundles "
        "(RBAC · ABAC · TRAC), edit policies, and compare baseline vs modified — "
        "no new inference required."
    )
