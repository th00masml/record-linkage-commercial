"""Audit 2: draw 100 pairs and apply the rules in PROTOCOL2.md."""
import json, math, os, random, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence import load
from evidence2 import all_pairs, evidence2

HERE = os.path.dirname(os.path.abspath(__file__))
EVER = {"G", "E", "S"}

def verdict(e):
    a, b, km = e["a"], e["b"], e["km"]
    far20 = km is not None and km > 20
    marks = a["visit_mark"] or b["visit_mark"]
    ma, mb = a["marriage"].strip(), b["marriage"].strip()
    if e["stems_differ"]:
        return "over_merge", "a_patronym"
    if (ma == "Ó" and mb in EVER) or (mb == "Ó" and ma in EVER):
        return "over_merge", "b_marital"
    if a["spouse_status"] and b["spouse_status"] and (e["partner_conflict"] or far20):
        return "over_merge", "c_two_households"
    if len(e["parallel_waves"]) >= 2 and far20 and not marks:
        return "over_merge", "d_parallel"
    if e["same_farm"] and e["by_diff"] is not None and e["by_diff"] <= 1:
        return "duplicate_row", ""
    if marks or (km is not None and km <= 5):
        return "double_listing", ""
    return "undetermined", ""

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return round((c - h) / d, 3), round((c + h) / d, 3)

def main():
    df, par = load()
    _, dups = all_pairs(df)
    rest = [tuple(x) for x in json.load(open(os.path.join(HERE, "remaining_pairs.json")))]
    sample = random.Random(0).sample(rest, 100)
    out = []
    for c in sample:
        e = evidence2(df, par, dups, *c)
        e["verdict"], e["rule"] = verdict(e)
        out.append(e)
    with open(os.path.join(HERE, "audit2_sample.jsonl"), "w") as f:
        for e in out: f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
    cnt = Counter(e["verdict"] for e in out); k = cnt["over_merge"]
    # stratified estimate over all within-wave positives (audit 1 rule: 35/50)
    N1, N2, p1, n1, p2, n2 = 859, 9429, 35/50, 50, k/100, 100
    W1, W2 = N1/(N1+N2), N2/(N1+N2)
    est = W1*p1 + W2*p2
    se = math.sqrt(W1**2*p1*(1-p1)/n1 + W2**2*p2*(1-p2)/n2)
    res = {"n": 100, "counts": dict(cnt), "rules_fired": dict(Counter(e["rule"] for e in out if e["rule"])),
           "over_merge_share": k/100, "wilson95": wilson(k, 100),
           "stratified_all_within_wave": {"estimate": round(est, 3), "ci95": [round(est-1.96*se, 3), round(est+1.96*se, 3)]}}
    json.dump(res, open(os.path.join(HERE, "rule_results2.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
