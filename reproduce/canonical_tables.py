"""Compute the rebuilt paper's canonical tables OFFLINE from outputs/canonical_rows/ dumps."""
import os
import json
import glob

D = os.path.join("outputs", "canonical_rows")


def load(name):
    return json.load(open(os.path.join(D, name), encoding="utf-8"))


def confusion(rows, deny):
    tp = fp = tn = fn = 0
    for x in rows:
        legit = x["is_legitimate"]
        allow = not deny(x)
        if not legit and not allow: tp += 1
        elif legit and not allow:   fp += 1
        elif legit and allow:       tn += 1
        else:                       fn += 1
    return tp, fp, tn, fn


def metr(rows, deny):
    tp, fp, tn, fn = confusion(rows, deny)
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0
    sf = fn/(tp+fn) if (tp+fn) else 0
    adm = tn/(tn+fp) if (tn+fp) else 0
    return dict(F1=round(f1,3), SecFail=round(sf,4), admit=round(100*adm,1), TP=tp, FP=fp, TN=tn, FN=fn)


# deny functions
d_e1  = lambda x: x["rbac_deny"] or x["abac_deny"] or x["tsphol_deny"]            # full det stack
d_e2  = lambda x: x["abac_deny"] or x["tsphol_deny"]                              # -RBAC
d_e3  = lambda x: x["tsphol_deny"]                                               # TRAC only
d_e4  = lambda x: x["llm_valid"] is False                                        # LLM only
d_full= lambda x: x["rbac_deny"] or x["abac_deny"] or (x["llm_valid"] is False) or x["tsphol_deny"]  # +LLM gate

VAL = ["gpt-4o", "gpt-5.4", "gemini-2.5-pro", "gpt-3.5-turbo-16k"]

print("="*78)
print("VALIDATION (agnostic engine, rescue=4.0)  n=6942")
print("="*78)
print(f'{"model":18s} | {"layer":12s} {"F1":>6} {"SecFail":>8} {"admit":>7}')
for m in VAL:
    rows = load(f"val_{m}_r4.json")
    for label, deny in [("E1 full-det", d_e1), ("E2 -RBAC", d_e2), ("E3 TRAC-only", d_e3),
                        ("E4 LLM-only", d_e4), ("FULL +LLMgate", d_full)]:
        r = metr(rows, deny)
        print(f'{m:18s} | {label:12s} {r["F1"]:>6.3f} {r["SecFail"]:>8.4f} {r["admit"]:>6.1f}%')
    print()

print("="*78)
print("CORROBORATED COVERAGE (validation, rescue 0 vs 4.0)")
print("="*78)
print(f'{"model":18s} | {"rescue":>6} {"det_SecF":>8} {"det_adm":>7} {"full_SecF":>9} {"full_adm":>8}')
for m in VAL:
    for rc in (0, 4):
        rows = load(f"val_{m}_r{rc}.json")
        de = metr(rows, d_e1); fu = metr(rows, d_full)
        print(f'{m:18s} | {rc:>6} {de["SecFail"]:>8.4f} {de["admit"]:>6.1f}% {fu["SecFail"]:>9.4f} {fu["admit"]:>7.1f}%')
    print()

print("="*78)
print("SELECTION (agnostic engine, full det stack, rescue=4.0)")
print("="*78)
print(f'{"model":18s} {"retr":7s} {"n":>5} | {"F1":>6} {"SecFail":>8} {"admit":>7}')
for fn in sorted(glob.glob(os.path.join(D, "sel_*.json"))):
    rows = json.load(open(fn, encoding="utf-8"))
    name = os.path.basename(fn)[4:-5]
    r = metr(rows, d_e1)
    print(f'{name:26s} {len(rows):>5} | {r["F1"]:>6.3f} {r["SecFail"]:>8.4f} {r["admit"]:>6.1f}%')

# RA-ICL paired on common cohort (gpt-5.4), joined on the (task, persona) row identity.
# NB: key by (task_idx, persona) -- keying by task_idx alone collapses the six personas.
print("\n--- RA-ICL paired (gpt-5.4, common cohort by (task_idx, persona)) ---")
none = {(r["task_idx"], r["persona"]): r for r in load("sel_gpt-5.4_none.json")}
for variant in ("bm25", "random"):
    var = load(f"sel_gpt-5.4_{variant}.json")
    ids = set((r["task_idx"], r["persona"]) for r in var)
    none_sub = [none[i] for i in ids if i in none]
    rn = metr(none_sub, d_e1); rv = metr(var, d_e1)
    print(f'  cohort={len(ids)} none.SecFail={rn["SecFail"]:.4f} {variant}.SecFail={rv["SecFail"]:.4f} '
          f'(ΔSecFail={rv["SecFail"]-rn["SecFail"]:+.4f})  none.admit={rn["admit"]:.1f}% {variant}.admit={rv["admit"]:.1f}%')
