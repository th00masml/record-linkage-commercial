"""Draw the 50-person sample and apply the pre-registered rules (PROTOCOL.md)."""
import json, math, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence import load, evidence

HERE = os.path.dirname(os.path.abspath(__file__))

def plausible(p, lo, hi):
    return p is None or p.get("age_gap") is None or lo <= p["age_gap"] <= hi

def anchored(s):
    f, m = s["father"], s["mother"]
    co = any(x and x.get("coresident") for x in (f, m))
    return co and plausible(f, 15, 70) and plausible(m, 14, 55) and s["patronym_ok"] is not False

def fails_link(s):
    return (not plausible(s["father"], 15, 70)) or (not plausible(s["mother"], 14, 55)) or s["patronym_ok"] is False

def coresident_parents(s):
    return {x["person"] for x in (s["father"], s["mother"]) if x and x.get("coresident")}

def verdict(e):
    a, b = e["a"], e["b"]
    dby = None if a["by"] is None or b["by"] is None else abs(a["by"] - b["by"])
    if anchored(a) and anchored(b) and not e["same_farm"] and not (coresident_parents(a) & coresident_parents(b)):
        return "over_merge"
    if e["same_farm"] and dby is not None and dby <= 1:
        return "duplicate_row"
    if (fails_link(a) != fails_link(b)) and e["km_ab"] is not None and e["km_ab"] <= 5 and dby is not None and dby <= 1:
        return "kin_link_error"
    return "undetermined"

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (c - h) / d, (c + h) / d

def main():
    df, par = load()
    cp = json.load(open(os.path.join(HERE, "conflict_pairs.json")))
    first = {}
    for c in cp:
        first.setdefault(c[0], c)
    persons = sorted(first)
    sample = random.Random(0).sample(persons, 50)
    out = []
    for p in sample:
        e = evidence(df, par, *first[p])
        e["verdict"] = verdict(e)
        out.append(e)
    with open(os.path.join(HERE, "audit_sample.jsonl"), "w") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
    from collections import Counter
    c = Counter(e["verdict"] for e in out)
    k = c["over_merge"]; lo, hi = wilson(k, 50)
    res = {"n": 50, "counts": dict(c), "over_merge_share": k/50, "wilson95": [round(lo, 3), round(hi, 3)]}
    json.dump(res, open(os.path.join(HERE, "rule_results.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
