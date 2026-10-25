"""Deterministic record-linkage baselines over the shown fields. Run first.
B1  = birth-year agreement only (very dumb).
B2  = a competent hand-built rule score (birthyear, parish, county, farm,
      partner, geographic proximity). Names are identical for every pair by
      construction, so they carry no signal -- this is the homonym problem.
"""
import json
import math
import os

from common import bootstrap_auc_ci, roc_auc

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
DERIVED = os.path.join(ROOT, "data/derived")
OUT = os.path.join(ROOT, "outputs")


def hav(a, b):
    try:
        la1, lo1, la2, lo2 = map(math.radians,
                                 [float(a["lat"]), float(a["lon"]),
                                  float(b["lat"]), float(b["lon"])])
    except Exception:
        return None
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(min(1, math.sqrt(d)))


def b1(a, b):
    dy = abs(a["birthyear"] - b["birthyear"])
    return 1.0 if dy == 0 else (0.5 if dy <= 1 else 0.0)


def b2(a, b):
    s = 0.0
    dy = abs(a["birthyear"] - b["birthyear"])
    s += 2.0 if dy == 0 else (1.0 if dy <= 1 else 0.0)
    if a["parish"] and a["parish"] == b["parish"]:
        s += 1.5
    if a["county"] and a["county"] == b["county"]:
        s += 0.5
    if a["farm_id"] and a["farm_id"] == b["farm_id"]:
        s += 1.0
    if a["partner_id"] and a["partner_id"] == b["partner_id"]:
        s += 2.0
    km = hav(a, b)
    if km is not None:
        s += max(0.0, 1.0 - km / 100.0)  # +1 if same place, decaying to 0 at 100km
    return s


def evaluate(name, fn, pairs):
    scores = [fn(p["a"], p["b"]) for p in pairs]
    labels = [p["label"] for p in pairs]
    auc = roc_auc(labels, scores)
    lo, hi = bootstrap_auc_ci(labels, scores)
    json.dump({"method": name, "auc": auc, "ci95": [lo, hi],
               "pid": [p["pid"] for p in pairs], "labels": labels, "scores": scores},
              open(os.path.join(OUT, f"scores_{name}.json"), "w"))
    print(f"{name:14} AUC={auc:.4f} [{lo:.4f},{hi:.4f}]")


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs = [json.loads(l) for l in open(os.path.join(DERIVED, "pairs.jsonl"))]
    evaluate("B1_birthyear", b1, pairs)
    evaluate("B2_rule", b2, pairs)


if __name__ == "__main__":
    main()
