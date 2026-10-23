import streamlit as st
import os
import json
import logging
from datetime import datetime
from typing import List, Any
from app.models.astra import AstraTask
from app.models.mcp import MCPPersona
from app.models.decision import DecisionResult
from app.services.llm_provider import LLMProvider
from app.services.prediction_service import PredictionService
from app.services.validation_service import ValidationService
from app.services.comparison_service import ComparisonService
from app.services.logger_service import LoggerService

from app.services.spiffe_registry_service import SpiffeRegistryService
from app.services.spiffe_allowlist_service import SpiffeAllowlistService
from app.services.rbac_service import RBACService
from app.services.tsphol_rule_service import TSPHOLRuleService
from app.services.mcp_attribute_service import MCPAttributeService
from app.services.decision_engine import DecisionEngine
from app.services.spiffe_workload_service import SpiffeWorkloadService
from app.services.reasoning_auditor import ReasoningAuditor

PREDICTION_LOG_DIR = os.path.join("datasets", "prediction_logs")
logger = logging.getLogger(__name__)


def _save_prediction_log(run_data: dict, task: AstraTask) -> str:
    """Save a single prediction lab run to a timestamped JSON log file."""
    os.makedirs(PREDICTION_LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ctx = run_data.get("experiment_context", {})
    mode_tag = "sel" if ctx.get("mode", "").startswith("Selection") else "val"
    import re
    persona_tag = ctx.get("caller_label", "unknown")
    persona_tag = re.sub(r'[^\w\-]', '_', persona_tag).strip('_').lower()
    filename = f"pred_{ts}_{mode_tag}_{persona_tag}.json"
    filepath = os.path.join(PREDICTION_LOG_DIR, filename)

    decision: DecisionResult = run_data["decision"]
    inference = run_data.get("inference")
    comparison = run_data.get("comparison")

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "context": ctx,
        "task": {
            "task_text": task.task,
            "match_tag": task.match_tag,
            "candidate_tools": task.candidate_tools,
            "candidate_mcp": task.candidate_mcp,
            "groundtruth_tools": task.groundtruth_tools,
            "groundtruth_mcp": task.groundtruth_mcp,
        },
        "decision": decision.model_dump(),
        "inference": _serialize_inference(inference),
        "comparison": comparison if isinstance(comparison, dict) else (
            comparison.model_dump() if hasattr(comparison, "model_dump") else str(comparison)
        ),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, default=str)

    logger.info("Prediction log saved to %s", filepath)
    return filepath


def _serialize_inference(inference) -> dict:
    """Serialize a SelectionResult or ValidationResult to a dict."""
    if inference is None:
        return {}
    if hasattr(inference, "model_dump"):
        return inference.model_dump()
    # Fallback for dataclass or plain object
    result = {}
    for attr in ["selected_mcp", "selected_tools", "justification",
                 "raw_output", "is_valid", "reason", "issues", "issue_codes",
                 "expected_domain", "actual_domain", "task_alignment_score",
                 "task_alignment_details", "validation_errors"]:
        if hasattr(inference, attr):
            result[attr] = getattr(inference, attr)
    return result

def render_prediction_lab(tasks: List[AstraTask], personas: List[MCPPersona]):
    st.title("🔮 Parallel reasoning lab")
    
    st.markdown("""
    Select an ASTRA task to trigger dual-mode reasoning and run the **Unified Decision Engine**:
    1. **LLM-ResM (Selection)**: Predict **exactly 3 pairs** of (Tool, MCP) from the catalog.
    2. **Validation**: Evaluate the **proposed candidate bundle** from the ASTRA dataset.
    """)
    
    # 5: Initialize Intent & Inference for Selection
    from app.services.intent_engine import IntentEngine
    from app.services.capability_inference_service import CapabilityInferenceService
    inference_svc = CapabilityInferenceService()
    intent_engine = IntentEngine(inference_svc=inference_svc)
    
    # Initialize core services using provider/model from sidebar
    from app.ui.llm_settings import get_llm_config, get_llm_config_error
    llm_cfg = get_llm_config()
    llm = LLMProvider(**llm_cfg)
    configuration_error = get_llm_config_error(llm_cfg) or llm.configuration_error
    if configuration_error:
        st.info(configuration_error)
    else:
        st.caption(f"Using {llm_cfg['provider']} / {llm_cfg['model']} from the sidebar.")
    predictor = PredictionService(llm, personas, intent_engine=intent_engine)
    validator = ValidationService(llm, personas)
    comparer = ComparisonService()
    logger = LoggerService()
    auditor = ReasoningAuditor(llm)
    
    # Initialize Policy Services
    registry_svc = SpiffeRegistryService()
    allowlist_svc = SpiffeAllowlistService(registry_service=registry_svc)
    rbac_svc = RBACService()
    tsphol_svc = TSPHOLRuleService()
    attribute_svc = MCPAttributeService()
    
    decision_engine = DecisionEngine(
        registry_svc=registry_svc,
        allowlist_svc=allowlist_svc,
        rbac_svc=rbac_svc,
        tsphol_svc=tsphol_svc,
        attribute_svc=attribute_svc,
        personas=personas
    )
    
    # Task Selection & Context Panel
    with st.container(border=True):
        st.subheader("📋 Context Initialization")
        
        # Caller simulation mapping
        registry = registry_svc.get_all()
        caller_options = []
        caller_map = {}
        for key, details in registry.items():
            disp_name = details.get("display_name", key)
            spiffe_id = details.get("spiffe_id", "")
            label = f"{disp_name} ({spiffe_id})"
            caller_options.append(label)
            caller_map[label] = {"spiffe_id": spiffe_id, "display_name": disp_name}
            
        col_c1, col_c2 = st.columns([1, 2])
        
        # SPIFFE Workload Identity Hook
        workload_svc = SpiffeWorkloadService()
        real_id, id_source = workload_svc.fetch_real_identity()
        
        with col_c1:
            # Map for simulation
            display_to_id = {}
            for label, data in caller_map.items():
                display_to_id[label] = data["spiffe_id"]
            
            sim_options = list(caller_map.keys())
            
            # Combine with Real Identity Option if available
            final_options = sim_options.copy()
            real_option = "🛡️ Use Real SPIFFE Identity"
            if real_id:
                final_options.insert(0, real_option)
                
            selected_caller_label = st.selectbox("Caller Identity (Authentication Source)", final_options, index=0 if final_options else None)
            
            if selected_caller_label == real_option:
                caller_spiffe_id = real_id
                # Find display name for real ID
                caller_display_name = "Unknown Persona"
                for label, data in caller_map.items():
                    if data["spiffe_id"] == real_id:
                        caller_display_name = data["display_name"]
                        break
                st.caption(f"Connected to SPIRE: `{real_id}`")
                id_source = "SPIRE Workload API"
            elif selected_caller_label:
                caller_spiffe_id = caller_map[selected_caller_label]["spiffe_id"]
                caller_display_name = caller_map[selected_caller_label]["display_name"]
                id_source = "Simulated"
            else:
                caller_spiffe_id = "spiffe://unknown"
                caller_display_name = "Unknown"
                id_source = "None"
        
            st.session_state["current_identity_source"] = id_source
            
        # MCP Filtering logic
        available_mcps = set()
        for t in tasks:
            if t.groundtruth_mcp:
                available_mcps.update(t.groundtruth_mcp)
        available_mcps = sorted(list(available_mcps))
        mcp_filter_options = ["All"] + available_mcps
        
        with col_c1:
            selected_mcp_filter = st.selectbox("Filter by MCP Server", mcp_filter_options)
            
        # Apply MCP filter
        if selected_mcp_filter == "All":
            mcp_filtered_tasks = tasks
        else:
            mcp_filtered_tasks = [t for t in tasks if t.groundtruth_mcp and selected_mcp_filter in t.groundtruth_mcp]
            
        if not mcp_filtered_tasks:
            st.warning("No tasks match the selected MCP Server filter.")
            return

        # Category Filtering logic
        available_tags = sorted(list(set([t.match_tag if t.match_tag else "Null" for t in mcp_filtered_tasks])))
        filter_options = ["All"] + available_tags
        
        with col_c1:
            selected_filter = st.selectbox("Filter by Task Category", filter_options)
        
        # Apply Category filter
        if selected_filter == "All":
            filtered_tasks = mcp_filtered_tasks
        else:
            filtered_tasks = [t for t in mcp_filtered_tasks if (t.match_tag if t.match_tag else "Null") == selected_filter]
            
        if not filtered_tasks:
            st.warning("No tasks match the selected Task Category filter.")
            return

        # 🎓 RA-ICL toggle (must come BEFORE the task index so we can
        # restrict the index range to the held-out test cohort).
        # Mode must be selected first so the RA-ICL widget can show
        # mode-appropriate options (selection-mode forbids in-domain
        # retrieval because the MCP label on every exemplar leaks the
        # answer; validation-mode permits it because the bundle already
        # declares the MCP).
        from app.ui.raicl_widget import render_k_selector
        split_info = None
        with col_c1:
            selected_mode = st.radio(
                "Experiment Mode",
                ["Selection (LLM-ResM)", "Validation"],
                horizontal=True,
                key="pred_mode_radio",
            )
            _mode_key = "selection" if selected_mode.startswith("Selection") else "validation"
        st.session_state["experiment_mode"] = selected_mode
        with col_c1:
            use_raicl = st.checkbox(
                "📚 Retrieval-augmented exemplars (RA-ICL, 70/30 split)", value=False,
                help="For each query, retrieve K relevant exemplars from the "
                     "70% train split and inject them into the prompt. The task "
                     "picker is restricted to the 30% held-out test cohort to "
                     "avoid evaluating on tasks that may have leaked into the pool.",
            )
            ra_choice = render_k_selector(
                key=f"pred_fs_k_{_mode_key}",  # remount when mode changes
                mode=_mode_key,
                disabled=not use_raicl,
            )

        if use_raicl:
            from app.services.split_service import load_or_build_split
            split_info = load_or_build_split(tasks, ratio=0.7, seed=42)
            test_fps = set(split_info.test_fingerprints)
            other_fps = set(split_info.other_fingerprints)
            allowed = test_fps | other_fps  # never train (would leak)
            # Filter the task list to evaluable tasks only
            from app.services.split_service import _task_fingerprint as _fp
            filtered_tasks = [t for t in filtered_tasks if _fp(t) in allowed]
            if not filtered_tasks:
                st.warning("No TEST or wrong/null tasks match the selected filters. "
                           "Either disable RA-ICL exemplars or widen the MCP/category filters.")
                return

        st.session_state["use_raicl"] = use_raicl
        st.session_state["ra_choice"] = ra_choice
        st.session_state["raicl_label"] = ra_choice.label
        st.session_state["ra_split"] = split_info

        with col_c2:
            label_suffix = " · TEST cohort only" if use_raicl else ""
            task_idx_in_filtered = st.number_input(
                f"Task Index (0-{len(filtered_tasks)-1}){label_suffix}",
                min_value=0, max_value=len(filtered_tasks)-1, value=0,
            )
            task = filtered_tasks[task_idx_in_filtered]
            st.info(f"**Task Description:**  \n{task.task}")
            st.caption(f"Category: {task.match_tag if task.match_tag else 'Null'}")

            if use_raicl and split_info is not None:
                cohort = split_info.classify(task)
                cohort_label = {"test": "🟩 TEST (held-out)", "other": "⬜ wrong/null"}.get(cohort, "🟦 train (?)")
                st.caption(
                    f"RA-ICL: {ra_choice.label} · cohort: {cohort_label} · "
                    f"train pool={len(split_info.train_fingerprints)} · test={len(split_info.test_fingerprints)}"
                )

            st.divider()
            
        # Do not show a previous model's result as the current configuration's output.
        context_key = (
            selected_mcp_filter, selected_filter, task_idx_in_filtered, selected_mode,
            llm_cfg["provider"], llm_cfg["model"],
        )
        if st.session_state.get("last_context_key") != context_key:
            st.session_state["last_run_data"] = None
            st.session_state["last_context_key"] = context_key
        
        # Dataset Context (ASTRA Candidates & Groundtruth)
        with st.expander("View Dataset Context (ASTRA Validation Details)", expanded=False):
            col_cand, col_gt = st.columns(2)
            with col_cand:
                if selected_mode == "Validation":
                    st.markdown("### 🧪 Candidate Bundle")
                    st.write("**Candidate MCPs:**")
                    st.json(task.candidate_mcp)
                    st.write("**Candidate Tools:**")
                    st.json(task.candidate_tools)
                else:
                    st.markdown("### 🧪 Candidate Bundle")
                    st.info("Hidden in Selection Mode to prevent leakage.")
                    
            with col_gt:
                st.markdown("### 🎯 Groundtruth")
                st.write("**Groundtruth MCPs:**")
                st.json(task.groundtruth_mcp)
                st.write("**Groundtruth Tools:**")
                st.json(task.groundtruth_tools)
    
    # Execution Button
    if st.button("🚀 Run Execution Pipeline", use_container_width=True,
                 disabled=bool(configuration_error)):
        if not llm.is_configured():
            st.error(llm.configuration_error)
        else:
            with st.spinner("Executing LLM Reasoning & Policy Engine..."):
                from app.models.selection import SelectionResult
                from app.models.validation import ValidationResult
                import json
                
                # Context Bundle for Audit
                experiment_context = {
                    "mode": selected_mode,
                    "provider": llm_cfg["provider"],
                    "model": llm_cfg["model"],
                    "caller_label": selected_caller_label,
                    "spiffe_id": caller_spiffe_id,
                    "id_source": id_source,
                    "mcp_filter": selected_mcp_filter,
                    "category_filter": selected_filter,
                    "task_index": task_idx_in_filtered,
                    "task_category": task.match_tag
                }

                # Exec results container
                run_data = {"experiment_context": experiment_context}

                # 1. Selection Pipeline
                if selected_mode.startswith("Selection"):
                    sel_pre_llm = decision_engine.pre_llm_check(caller_spiffe_id, None, None)
                    if sel_pre_llm["passed"]:
                        # Build exemplars on-demand if RA-ICL enabled
                        sel_exemplars = None
                        if st.session_state.get("use_raicl") and st.session_state.get("ra_split") is not None:
                            from app.services.exemplar_retriever import ExemplarRetriever
                            ra_split = st.session_state["ra_split"]
                            train_tasks = ra_split.filter_train(tasks)
                            _choice = st.session_state["ra_choice"]
                            retriever = ExemplarRetriever(
                                train_tasks,
                                k=_choice.resolve_k(len(train_tasks)),
                                strategy=_choice.strategy,
                                seed=42,
                                pad_cross_domain=_choice.pad_cross_domain,
                            )
                            sel_exemplars = retriever.get(task)
                        selection = predictor.run_selection(task, exemplars=sel_exemplars)
                        sel_comparison = comparer.compare(task.groundtruth_mcp, task.groundtruth_tools, selection.selected_mcp, selection.selected_tools)
                        sel_raw = {}
                        try:
                            if selection.raw_output: sel_raw = json.loads(selection.raw_output)
                        except: pass
                        sel_decision = decision_engine.evaluate(sel_pre_llm, caller_spiffe_id, selection.selected_mcp, selection.selected_tools, sel_raw, task.task, mode="selection", mcp_filter=selected_mcp_filter)
                        sel_decision.llm_executed = True
                        sel_decision.llm_output = {"selected_mcp": selection.selected_mcp, "selected_tools": selection.selected_tools, "justification": selection.justification}
                    else:
                        selection = SelectionResult(selected_mcp=[], selected_tools=[], justification="Skipped due to Pre-LLM block", raw_output=None)
                        sel_comparison = comparer.compare(task.groundtruth_mcp, task.groundtruth_tools, [], [])
                        sel_decision = decision_engine.evaluate(sel_pre_llm, caller_spiffe_id, [], [], {}, task.task)
                        sel_decision.llm_executed = False
                    
                    run_data.update({
                        "decision": sel_decision,
                        "inference": selection,
                        "comparison": sel_comparison
                    })

                # 2. Validation Pipeline
                elif selected_mode == "Validation":
                    val_pre_llm = decision_engine.pre_llm_check(caller_spiffe_id, task.candidate_mcp, task.candidate_tools)
                    if val_pre_llm["passed"]:
                        val_exemplars = None
                        if st.session_state.get("use_raicl") and st.session_state.get("ra_split") is not None:
                            from app.services.exemplar_retriever import ExemplarRetriever
                            ra_split = st.session_state["ra_split"]
                            train_tasks = ra_split.filter_train(tasks)
                            _choice = st.session_state["ra_choice"]
                            retriever = ExemplarRetriever(
                                train_tasks,
                                k=_choice.resolve_k(len(train_tasks)),
                                strategy=_choice.strategy,
                                seed=42,
                                pad_cross_domain=_choice.pad_cross_domain,
                            )
                            val_exemplars = retriever.get(task)
                        validation = validator.run_validation(task, exemplars=val_exemplars)
                        val_comparison = comparer.compare(task.groundtruth_mcp, task.groundtruth_tools, task.candidate_mcp, task.candidate_tools)
                        val_raw = {}
                        try:
                            if validation.raw_output: val_raw = json.loads(validation.raw_output)
                        except: pass
                        val_decision = decision_engine.evaluate(val_pre_llm, caller_spiffe_id, task.candidate_mcp, task.candidate_tools, val_raw, task.task, mode="validation", mcp_filter=selected_mcp_filter)
                        val_decision.llm_executed = True
                        val_decision.llm_output = {
                            "is_valid": validation.is_valid, 
                            "reason": validation.reason, 
                            "issues": validation.issues,
                            "issue_codes": validation.issue_codes,
                            "expected_domain": validation.expected_domain,
                            "actual_domain": validation.actual_domain,
                            "task_alignment_score": validation.task_alignment_score,
                            "task_alignment_details": validation.task_alignment_details
                        }
                    else:
                        validation = ValidationResult(is_valid=False, reason="Skipped due to Pre-LLM block", issues=["SKIPPED"], raw_output=None)
                        val_comparison = comparer.compare(task.groundtruth_mcp, task.groundtruth_tools, task.candidate_mcp, task.candidate_tools)
                        val_decision = decision_engine.evaluate(val_pre_llm, caller_spiffe_id, task.candidate_mcp, task.candidate_tools, {}, task.task)
                        val_decision.llm_executed = False
                    
                    run_data.update({
                        "decision": val_decision,
                        "inference": validation,
                        "comparison": val_comparison
                    })
                
                st.session_state["last_run_data"] = run_data

                # Persist log to disk
                try:
                    log_path = _save_prediction_log(run_data, task)
                    st.success(f"📄 Prediction log saved to `{log_path}`")
                except Exception as e:
                    st.warning(f"⚠️ Could not save prediction log: {e}")

    # --- Persistent Result Rendering ---
    last_run = st.session_state.get("last_run_data")
    if last_run:
        decision = last_run["decision"]
        inference = last_run["inference"]
        comparison = last_run["comparison"]
        ctx = last_run["experiment_context"]
        
        # Display Architecture
        mode_tag = "selection" if ctx["mode"].startswith("Selection") else "validation"
        title = "🧠 Research Mode: Selection (LLM-ResM)" if mode_tag == "selection" else "🛡️ Research Mode: Validation (Meta-Level)"
        st.header(title)
        
        # Phase 1: Context
        _render_phase_1(decision, caller_display_name, mode=mode_tag)
        st.divider()
        
        # Phase 2: Generation
        if decision.llm_executed:
            _render_phase_2(inference, mode_tag, task=task)
        else:
            st.warning("⚠️ GENERATION LAYER SKIPPED: Pre-LLM checks failed.")
        st.divider()
        
        # Phase 3: Logic
        _render_phase_3(decision, comparison, mode=mode_tag, inference=inference)
        
        # Details & Auditor
        with st.expander("📄 Raw Model Outputs & Details", expanded=False):
            st.subheader("📋 Context Initialization Audit")
            st.json(ctx)
            
            st.divider()
            if ctx["mode"].startswith("Selection"):
                st.subheader("Selection Inference Details")
                st.write("**Inference Justification:**", inference.justification)
                st.write("**Comparison Result:**")
                st.json(comparison.model_dump())
            else:
                st.subheader("Validation Inference Details")
                st.write("**Inference Reason:**", inference.reason)
                st.write("**Detected Issues:**")
                st.json(inference.issues)
            
            st.subheader("Detailed Logical Predicates")
            st.json(decision.context)

        # --- Post-Reasoning Auditor ---
        st.divider()
        st.markdown("### 🔍 Run Assessment & Recommendations")
        
        state_key = f"audit_{ctx['mode']}_{ctx['task_index']}"
        if state_key not in st.session_state:
            st.session_state[state_key] = None
        
        if st.button("🚀 Generate Logic Post-Mortem & Strategic Advice", key=f"btn_{state_key}",
                     disabled=bool(configuration_error)):
            with st.spinner("LLM Auditor analyzing execution trace..."):
                assessment = auditor.generate_assessment(
                    task=task.task,
                    metadata=ctx,
                    decision_data=decision.model_dump(),
                    benchmark_status=comparison.status
                )
                st.session_state[state_key] = assessment
        
        report = st.session_state[state_key]
        if report:
            with st.container(border=True):
                st.markdown(f"#### 📊 Auditor Summary: {report.get('summary')}")
                for section in report.get("sections", []):
                    with st.expander(f"📌 {section.get('title')}", expanded=True):
                        st.markdown(section.get("content"))
                if report.get("recommendations"):
                    st.markdown("---")
                    st.markdown("**💡 Strategic Recommendations:**")
                    for rec in report.get("recommendations", []):
                        st.success(f"- {rec}")

def _render_authorization_trace(decision: DecisionResult):
    """
    Helper to render the RBAC and ABAC status and detailed trace.
    Both layers enforce independently in the pipeline.
    """
    rbac_status = decision.evaluation_states.get("rbac", "NOT_EVALUATED")
    abac_status = decision.evaluation_states.get("abac", "NOT_EVALUATED")

    with st.expander("⚖️ View Authorization Trace (RBAC → ABAC)"):
        # Display Subject Attributes
        from app.services.spiffe_registry_service import SpiffeRegistryService
        registry = SpiffeRegistryService().get_all()
        persona_data = next((v for k, v in registry.items() if v.get("spiffe_id") == decision.spiffe_id), {})
        attrs = persona_data.get("attributes", {})
        if attrs:
            col1, col2, col3 = st.columns(3)
            col1.write(f"**Dept:** {attrs.get('department', 'Unknown')}")
            col2.write(f"**Clearance:** {attrs.get('clearance_level', 'L1')}")
            col3.write(f"**Trust:** {attrs.get('trust_score', 0.5):.1f}")
            st.divider()

        rbac_audit = decision.context.get("rbac_evaluation", {})
        if rbac_audit:
            st.markdown(f"**Overall RBAC Result:** `{rbac_audit.get('decision')}` | Policy: `Persona Role`")
            st.caption(rbac_audit.get("reason", ""))
            
            # Detailed Tool Breakdown
            rbac_trace = rbac_audit.get("rbac_trace", [])
            if rbac_trace:
                import pandas as pd
                st.markdown("**Per-Tool Authorization Trace:**")
                rdf = pd.DataFrame(rbac_trace)
                rdf.columns = ["Tool", "MCP", "Decision", "Matched Rule", "Rationale"]
                st.table(rdf)
        
        st.divider()
        abac = decision.context.get("abac_baseline", {})
        if abac:
            abac_dec = abac.get('decision', 'N/A')
            abac_icon = "✅" if abac_dec == "ALLOW" else "❌"
            st.markdown(f"**ABAC Result:** {abac_icon} `{abac_dec}` | Rule: `{abac.get('matched_rule')}`")
            if abac_dec == "DENY":
                st.caption("⚡ ABAC enforced this denial directly — pipeline short-circuited.")
            
            # ABAC Attribute Mapping Detail (6D Detailed Trace)
            trace_steps = abac.get("reasoning_trace", {}).get("logic_steps", [])
            if trace_steps:
                import pandas as pd
                st.markdown("**Attribute Authority Evidence:**")
                
                # Transform logic steps into display rows
                rows = []
                for s in trace_steps:
                    rows.append({
                        "Condition": s.get("condition"),
                        "Value": s.get("actual"),
                        "Decision": s.get("decision", "ALLOW"),
                        "Rule ID": s.get("rule_id", "baseline")
                    })
                
                st.table(pd.DataFrame(rows))
            
            if abac.get("failure_reason"):
                st.error(f"ABAC Denial Rationale: {abac.get('failure_reason')}")
        else:
            st.info("No ABAC status available.")

def _render_phase_1(decision: DecisionResult, caller_display_name: str, mode: str = "validation"):
    """
    PHASE 1: PRE-LLM CONTEXT & IDENTITY
    """
    st.markdown("### 🏛️ Phase I: Context & Identity (Pre-LLM)")
    with st.container(border=True):
        st.write(f"**Subject Caller:** {caller_display_name}")
        st.code(decision.spiffe_id)
        
        id_source = st.session_state.get("current_identity_source", "Unknown")
        if "api" in id_source.lower() or "real" in id_source.lower():
            st.success(f"**Identity Source:** {id_source} | 🛡️ Real SPIFFE ID")
        else:
            st.info(f"**Identity Source:** {id_source} (Simulated)")
            
        id_status = decision.evaluation_states.get("identity", "NOT_EVALUATED")
        tr_status = decision.evaluation_states.get("transport", "NOT_EVALUATED")
        
        if mode == "selection":
            c1, c2 = st.columns(2)
            c1.metric("Identity Verified", "ALLOW" if id_status == "ALLOW" else "DENY")
            c2.metric("mTLS Transport", "ALLOW" if tr_status == "ALLOW" else "DENY")
        else:
            rbac_status = decision.evaluation_states.get("rbac", "NOT_EVALUATED")
            abac_status = decision.evaluation_states.get("abac", "NOT_EVALUATED")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Identity Verified", "ALLOW" if id_status == "ALLOW" else "DENY")
            c2.metric("mTLS Transport", "ALLOW" if tr_status == "ALLOW" else "DENY")
            c3.metric("RBAC (Identity)", rbac_status)
            c4.metric("ABAC (Attributes)", abac_status)
            
            if rbac_status == "DENY":
                st.error(f"🛑 RBAC Denial: {decision.reason}")
            elif abac_status == "DENY":
                st.error(f"🛑 ABAC Denial: {decision.reason}")
            
            _render_authorization_trace(decision)

def _render_phase_2(result: Any, mode: str, task: Any = None):
    """
    PHASE 2: LLM INFERENCE (GENERATION LAYER)
    Displays the LLM logic hypotheses and alignment scores.
    """
    st.markdown("### 🧠 Phase II: LLM Generation (Inference)")
    with st.container(border=True):
        if mode == "selection":
            st.subheader("Logic Hypothesis (Tools Selection)")
            st.write("**Predicted MCPs:**")
            st.json(result.selected_mcp)
            st.write("**Predicted Tools:**")
            st.json(result.selected_tools)
            
            st.markdown(f"**Model Justification:** {result.justification}")
            
        else:
            st.subheader("Meta-Level Logic Verification")
            st.write("**Grounded Bundle for Validation:**")
            st.json({"MCPs": task.candidate_mcp, "Tools": task.candidate_tools})
            
            v_out = result.model_dump() if hasattr(result, "model_dump") else result
            
            c1, c2 = st.columns(2)
            alignment_score = v_out.get("task_alignment_score", 0.0)
            eval_eval = "✅" if alignment_score > 0 else "❌" # Simulated indicator
            c1.metric(f"Alignment Score {eval_eval}", f"{alignment_score:.2f}")
            
            is_valid = v_out.get("is_valid", False)
            c2.metric("Logical Validity", "VALID" if is_valid else "INVALID")
            
            if v_out.get("issue_codes"):
                st.write("**Detected Issue Codes:**")
                st.warning(", ".join(v_out.get("issue_codes", [])))
            
            st.markdown(f"**Validator Reasoning:** {v_out.get('reason')}")
            
            components = v_out.get("task_alignment_details", {})
            if components:
                with st.expander("📊 Alignment Breakdown (Weighted Reasoning)"):
                    st.json(components)

def _render_phase_3(decision: DecisionResult, comparison: Any, mode: str = "validation", inference: Any = None):
    """
    PHASE 3: VERIFIED LOGIC TRACE (POST-LLM)
    Displays the final grounding, TRAC audit, and verified decision.
    """
    st.markdown("### 🚦 Phase III: Verified Logic Trace (Post-LLM)")
    with st.container(border=True):
        st.markdown("#### D. Fact Extraction & Audit")
        audit_data = decision.context.get("tool_audit", [])
        intent = decision.context.get("intent_decomposition", {})
        
        with st.expander("🔍 Domain-Aware Predicate Audit", expanded=True):
            if audit_data:
                import pandas as pd
                df = pd.DataFrame(audit_data)
                # Filter caps
                from app.services.domain_capability_ontology import DomainCapabilityOntology
                df["capabilities"] = df["capabilities"].apply(lambda caps: [c for c in caps if DomainCapabilityOntology.is_concrete(c)])
                df_disp = df[["tool", "source", "actions", "capabilities", "notes"]].copy()
                df_disp.columns = ["Tool Name", "Classification Authority", "Action Types", "Concrete Caps", "Grounding Rationale"]
                st.table(df_disp)
                st.caption("💡 **Classification Authority**: Where the mapping comes from (Curated, Heuristic, or Catalog).")
                st.caption("💡 **Grounding Rationale**: Detailed evidence for the identified capabilities.")
            else:
                st.info("No tool audit data available.")
        
        if mode == "selection":
            st.divider()
            st.markdown("#### E.2 Governance Pipeline (RBAC → ABAC → TRAC)")
            st.caption(
                "Each layer enforces independently. If any layer denies, "
                "the pipeline short-circuits — downstream layers are not reached."
            )
            rbac_status = decision.evaluation_states.get("rbac", "NOT_EVALUATED")
            abac_status = decision.evaluation_states.get("abac", "NOT_EVALUATED")
            tsphol_quick = decision.evaluation_states.get("tsphol", "NOT_EVALUATED")

            c1, c2, c3 = st.columns(3)
            # RBAC
            rbac_icon = "✅" if rbac_status == "ALLOW" else ("❌" if rbac_status == "DENY" else "⏭️")
            c1.metric(f"{rbac_icon} RBAC (Identity)", rbac_status)
            # ABAC
            abac_icon = "✅" if abac_status == "ALLOW" else ("❌" if abac_status == "DENY" else "⏭️")
            c2.metric(f"{abac_icon} ABAC (Attributes)", abac_status)
            # TRAC
            tsphol_icon = "✅" if tsphol_quick == "ALLOW" else ("❌" if tsphol_quick == "DENY" else "⏭️")
            c3.metric(f"{tsphol_icon} TRAC (Logic)", tsphol_quick)

            if rbac_status == "DENY":
                st.error(f"🛑 **RBAC Denial** — {decision.reason}")
                st.caption("Pipeline short-circuited at RBAC. ABAC and TRAC were not evaluated.")
            elif abac_status == "DENY":
                st.error(f"🛑 **ABAC Denial** — {decision.reason}")
                st.caption("Pipeline short-circuited at ABAC. TRAC was not evaluated.")
            
            _render_authorization_trace(decision)

        st.divider()
        st.markdown("#### E. TRAC Final Authority")
        tsphol_status = decision.evaluation_states.get("tsphol", "NOT_EVALUATED")
        summary = decision.context.get("tsphol_summary", {})
        
        if tsphol_status == "NOT_EVALUATED":
            # Determine why TRAC was not reached
            rbac_s = decision.evaluation_states.get("rbac", "NOT_EVALUATED")
            abac_s = decision.evaluation_states.get("abac", "NOT_EVALUATED")
            if rbac_s == "DENY":
                st.info("TRAC not reached — pipeline short-circuited at **RBAC**.")
            elif abac_s == "DENY":
                st.info("TRAC not reached — pipeline short-circuited at **ABAC**.")
            else:
                st.info("TRAC not reached (Pre-LLM Block).")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Rules Evaluated", summary.get("evaluated_rules", 0))
            c2.metric("Rules Triggered", summary.get("triggered_rules", 0))
            c3.metric("Math Certainty", f"{summary.get('certainty', 1.0):.0%}")
            
            with st.expander("🔍 Detailed Logic Evaluation Trace"):
                l_trace = decision.context.get("tsphol_logic_trace", [])
                for entry in l_trace:
                    icon = "✅ Pass" if entry["passed"] else "❌ Triggered"
                    st.write(f"{icon}: `{entry['rule']}`")
                    st.caption(f"↳ {entry['reason']}")
            
            # 6D: Redundant positive findings removed as per user request
            pass

        st.divider()
        st.markdown("#### F. Final Result")
        color_map = {"ALLOW": "green", "DENY": "red"}
        bg_color = color_map.get(decision.final_decision, "gray")
        display_text = decision.final_decision

        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 8px; text-align: center;">
            <h2 style="color: white; margin: 0;">{display_text}</h2>
        </div>
        """, unsafe_allow_html=True)
        st.info(f"**Conclusion:** {decision.reason}")
        
        # Benchmark Evaluation Section (Moved to end for clarity)
        st.divider()
        st.markdown("### 📊 Benchmark Assessment (Reference Only)")
        st_color = {"exact_match": "green", "partial_match": "orange", "mismatch": "red"}.get(comparison.status, "gray")
        st.markdown(f"**Groundtruth Alignment Status:** :{st_color}[{comparison.status.upper()}]")
        st.caption("⚠️ **Research Disclaimer**: This assessment reflects a direct comparison against the ASTRA dataset. This data is **NOT** shared with the LLM and is not used in the logical decision pipeline. It is provided purely for benchmark analysis.")

        with st.expander("View Full Pipeline Trace"):
            for step in decision.trace:
                if "❌" in step or "DENY" in step:
                    st.error(step)
                elif "⚠️" in step or "FLAG" in step:
                    st.warning(step)
                elif "✅" in step or "ALLOW" in step:
                    st.success(step)
                else:
                    st.info(step)
