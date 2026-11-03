"""Assemble the FULL cross-vendor panel (Table 3) — canonical scoring, n=6,942.

Reuses the paper's canonical OpenAI rows (outputs/canonical_rows/val_gpt-*_r4.json) and
scores the 3 fresh non-OpenAI logs (datasets/llm_inference_logs/*_<model>_validation.json)
through the IDENTICAL replay path (RESCUE=4.0, baseline_policies) — exactly like
outputs/canonical_rebuild.py + canonical_tables.py. Prints a combined table, writes
outputs/crossvendor_full.json, and emits LaTeX rows for tab:cross_vendor.

Run after the 3 jobs finish:  python outputs/assemble_crossvendor.py
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CANON = os.path.join("outputs", "canonical_rows")
LL = os.path.join("datasets", "llm_inference_logs")

# Display metadata: label -> (vendor, year, kind)
OPENAI = [
    ("gpt-3.5-turbo-16k", "OpenAI", 2023),
    ("gpt-4o",            "OpenAI", 2024),
    ("gpt-5.4",           "OpenAI", 2026),
]
# (label, provider, model-slug-in-filename, vendor, year)
NEW = [
    ("claude-opus-4.8",   "anthropic", "claude-opus-4-8",   "Anthropic", 2026),
    ("claude-sonnet-4.6", "anthropic", "claude-sonnet-4-6", "Anthropic", 2026),
    ("gemini-2.5-pro",    "google",    "gemini-2-5-pro",    "Google",    2026),
]

FIELDS = ("persona", "task_idx", "domain", "match_tag", "is_legitimate",
          "rbac_deny", "abac_deny", "tsphol_deny", "tsphol_rule", "llm_valid",
          "contains_write", "hard_missing", "tsphol_advisory_rules")

d_e1   = lambda x: x["rbac_deny"] or x["abac_deny"] or x["tsphol_deny"]
d_e4   = lambda x: x["llm_valid"] is False
d_full = lambda x: x["rbac_deny"] or x["abac_deny"] or (x["llm_valid"] is False) or x["tsphol_deny"]


def metr(rows, deny):
    tp = fp = tn = fn = 0
    for x in rows:
        legit = x["is_legitimate"]; allow = not deny(x)
        if not legit and not allow: tp += 1
        elif legit and not allow:   fp += 1
        elif legit and allow:       tn += 1
        else:                       fn += 1
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0
    sf = fn/(tp+fn) if (tp+fn) else 0
    adm = tn/(tn+fp) if (tn+fp) else 0
    return dict(f1=round(f1, 3), secfail=round(sf, 4), admit=round(100*adm, 1))


def rows_from_canon(model):
    p = os.path.join(CANON, f"val_{model}_r4.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def rows_from_fresh_log(model_slug):
    cands = sorted(glob.glob(os.path.join(LL, f"*_{model_slug}_validation.json")))
    if not cands:
        return None, None
    log = cands[-1]  # newest timestamp
    from app.loaders.astra_loader import load_astra_dataset
    from app.services import replay_service as rs
    from app.services import tool_relevance as trel
    trel.RESCUE_RELEVANCE = 4.0
    tasks = load_astra_dataset(os.path.join("datasets", "astra_03_tools.json"))
    robjs, _, _ = rs.replay_experiment(log, tasks, experiment="E1", limit=None,
                                       policies=rs.baseline_policies())
    rows = [{f: getattr(x, f, None) for f in FIELDS} for x in robjs]
    return rows, log


def summarize(rows):
    return {"n": len(rows), "e1": metr(rows, d_e1), "e4": metr(rows, d_e4), "full": metr(rows, d_full)}


def main():
    out = {"openai": {}, "nonopenai": {}, "e1_floor": None}
    print("=" * 92)
    print(f'{"model":20s} {"vendor":10s} {"n":>5} | {"E4 F1":>6} {"E4 SecF":>8} {"E4 Adm":>7} | '
          f'{"FULL F1":>7} {"FULL SecF":>9} {"FULL Adm":>8}')
    print("-" * 92)
    for label, vendor, year in OPENAI:
        rows = rows_from_canon(label)
        if rows is None:
            print(f"{label:20s} {vendor:10s}  MISSING canonical rows"); continue
        s = summarize(rows); out["openai"][label] = s; out["e1_floor"] = s["e1"]
        print(f'{label:20s} {vendor:10s} {s["n"]:>5} | {s["e4"]["f1"]:>6.3f} {s["e4"]["secfail"]:>8.4f} '
              f'{s["e4"]["admit"]:>6.1f}% | {s["full"]["f1"]:>7.3f} {s["full"]["secfail"]:>9.4f} {s["full"]["admit"]:>7.1f}%')
    print("-" * 92)
    for label, prov, slug, vendor, year in NEW:
        rows, log = rows_from_fresh_log(slug)
        if rows is None:
            print(f"{label:20s} {vendor:10s}  (log not ready)"); continue
        s = summarize(rows); s["log"] = log; out["nonopenai"][label] = s
        print(f'{label:20s} {vendor:10s} {s["n"]:>5} | {s["e4"]["f1"]:>6.3f} {s["e4"]["secfail"]:>8.4f} '
              f'{s["e4"]["admit"]:>6.1f}% | {s["full"]["f1"]:>7.3f} {s["full"]["secfail"]:>9.4f} {s["full"]["admit"]:>7.1f}%')
    if out["e1_floor"]:
        f = out["e1_floor"]
        print("-" * 92)
        print(f'E1 deterministic floor (model-independent): F1={f["f1"]}  SecFail={f["secfail"]}  Admit={f["admit"]}%')

    json.dump(out, open(os.path.join("outputs", "crossvendor_full.json"), "w"), indent=2)
    print("\nSaved -> outputs/crossvendor_full.json")


if __name__ == "__main__":
    main()
