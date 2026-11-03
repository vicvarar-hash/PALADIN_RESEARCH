"""Phase 1 of the paper re-baseline: regenerate ALL decision rows on the CURRENT agnostic engine.

For each (model, mode[, retrieval]) we replay the cached LLM-inference log through the live
deterministic stack (RBAC / ABAC / TRAC, agnostic {domain}:{action}, corroborated coverage) and
persist the per-row decision fields. Every paper table (confusion matrices, F1/SecFail/admission,
per-layer marginals, per-domain, per-persona, write/read, RA-ICL, corroborated coverage) is then
computed OFFLINE from these row dumps — no re-replaying.

Validation models are dumped at BOTH rescue=0 and rescue=4.0 (to document corroborated coverage);
everything else at the production default rescue=4.0.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.loaders.astra_loader import load_astra_dataset
from app.services import replay_service as rs
from app.services import tool_relevance as trel

LL = os.path.join("datasets", "llm_inference_logs")
OUTDIR = os.path.join("outputs", "canonical_rows")

VAL = [
    ("gpt-4o",            "20260613005419_gpt-4o_validation.json"),
    ("gpt-5.4",           "20260612191843_gpt-5-4_validation.json"),
    ("gemini-2.5-pro",    "20260708143259_gemini-2-5-pro_validation.json"),
    ("gpt-3.5-turbo-16k", "20260613141204_gpt-35-turbo-16k_validation.json"),
]
SEL = [
    ("gpt-4o",          "none",   "20260529112541_gpt-4o_selection.json"),
    ("gpt-3.5-turbo",   "none",   "20260505161002_gpt-3-5-turbo_selection.json"),
    ("claude-opus-4-8", "none",   "20260612103402_claude-opus-4-8_selection.json"),
    ("gpt-5.4",         "none",   "20260613105137_gpt-5-4_selection.json"),
    ("gpt-5.4",         "bm25",   "20260613165151_gpt-5-4_selection_ra-bm25-k10000.json"),
    ("gpt-5.4",         "random", "20260614030129_gpt-5-4_selection_ra-random-any-k10000.json"),
]

FIELDS = ("persona", "task_idx", "domain", "match_tag", "is_legitimate",
          "rbac_deny", "abac_deny", "tsphol_deny", "tsphol_rule", "llm_valid",
          "contains_write", "hard_missing", "tsphol_advisory_rules")


def dump(rows, path):
    out = []
    for x in rows:
        out.append({f: getattr(x, f, None) for f in FIELDS})
    json.dump(out, open(path, "w"), separators=(",", ":"))


def quick(rows, with_llm):
    legit = [x for x in rows if x.is_legitimate]
    illeg = [x for x in rows if not x.is_legitimate]
    def deny(x):
        d = x.rbac_deny or x.abac_deny or x.tsphol_deny
        return d or (with_llm and x.llm_valid is False)
    sf = sum(1 for x in illeg if not deny(x)) / len(illeg) if illeg else 0
    la = sum(1 for x in legit if not deny(x)) / len(legit) if legit else 0
    return sf, la


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    tasks = load_astra_dataset(os.path.join("datasets", "astra_03_tools.json"))

    print("=== VALIDATION (det stack SecFail/admit @rescue) ===", flush=True)
    for model, fn in VAL:
        for rescue in (4.0, 0.0):
            trel.RESCUE_RELEVANCE = rescue
            rows, _, _ = rs.replay_experiment(os.path.join(LL, fn), tasks, experiment="E1",
                                              limit=None, policies=rs.baseline_policies())
            dump(rows, os.path.join(OUTDIR, f"val_{model}_r{int(rescue)}.json"))
            sf, la = quick(rows, with_llm=False)
            print(f"  {model:18s} r={rescue:<4} det SecFail={sf:.4f} admit={100*la:.1f}%  (n={len(rows)})", flush=True)

    print("\n=== SELECTION (det stack @rescue=4.0) ===", flush=True)
    trel.RESCUE_RELEVANCE = 4.0
    for model, retr, fn in SEL:
        rows, _, _ = rs.replay_experiment(os.path.join(LL, fn), tasks, experiment="E1",
                                          limit=None, policies=rs.baseline_policies())
        dump(rows, os.path.join(OUTDIR, f"sel_{model}_{retr}.json"))
        sf, la = quick(rows, with_llm=False)
        print(f"  {model:18s} {retr:7s} det SecFail={sf:.4f} admit={100*la:.1f}%  (n={len(rows)})", flush=True)

    print("\nSaved per-row dumps -> outputs/canonical_rows/", flush=True)


if __name__ == "__main__":
    main()
