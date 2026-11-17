"""Task 3 (gray-zone-triage plan): calibrate B2 and freeze the gray-zone thresholds.

Uses the CALIBRATION split only. Test rows are dropped immediately after
loading and nothing is computed on them.

Steps (plan Task 3, Frozen parameters, Amendment 1):
1. Calibration sample: the distinct anchor records (anchor_rid) of split "cal",
   sorted ascending, shuffled with random.Random(0); the first 3,000 are taken
   with all their candidate pairs.
2. Score every pair with the frozen rule src/baseline.py::b2 on records from
   records.jsonl (partner_id already blanked by Task 2, Amendment 1).
3. Calibrate: unpenalised logistic regression label ~ b2 score
   (sklearn LogisticRegression, penalty=None), p = sigma(b0 + b1 * score).
4. Thresholds over the distinct calibrated values p_1 < ... < p_m:
   T_U = smallest p_j such that among pairs with p >= p_j the share of
         non-matches (false-match rate of auto-accepted links) <= alpha;
   T_L = largest p_j such that among pairs with p < p_j the share of matches
         (missed-match rate of auto-rejected pairs) <= beta.
   Decision regions: p >= T_U auto-accept; p < T_L auto-reject;
   T_L <= p < T_U gray zone (clerical review / second stage).
   If no admissible T_U exists, T_U = +inf (nothing auto-accepted); if no
   admissible T_L exists, T_L = p_1 (nothing auto-rejected).
   If T_L >= T_U there is no gray zone: falsification criterion 5 applies.
5. Write results/thresholds.json. Commit it before any test label or LLM
   output is read (plan Verification).

Assumptions: B2's score is a deterministic function of the record pair, so all
pairs with the same score share one calibrated p and fall in the same region.
Monotone calibration (b1 > 0) is checked; if b1 <= 0 the script stops.
"""
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(PROJ)
DERIVED = os.path.join(PROJ, "input", "data", "derived")
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(REPO, "src"))
from baseline import b2  # noqa: E402  frozen rule

SEED = 0
N_CAL_ANCHORS = 3000
ALPHA = 0.05
BETA = 0.05


def thresholds(p, y, alpha, beta):
    """T_U and T_L over distinct calibrated values (see module docstring)."""
    vals = np.unique(p)                       # ascending
    order = np.argsort(p)
    ps, ys = p[order], y[order]
    n = len(ps)
    # suffix counts for p >= v, prefix counts for p < v
    idx = np.searchsorted(ps, vals, side="left")      # first index with p >= v
    pos_cum = np.concatenate([[0], np.cumsum(ys)])    # positives among first k
    tot_pos = pos_cum[-1]
    t_u = math.inf
    for v, k in zip(vals, idx):
        m = n - k
        fm = (m - (tot_pos - pos_cum[k])) / m          # non-matches among p >= v
        if fm <= alpha:
            t_u = float(v)
            break
    t_l = float(vals[0])
    for v, k in zip(vals[::-1], idx[::-1]):
        if k == 0:
            continue
        mm = pos_cum[k] / k                            # matches among p < v
        if mm <= beta:
            t_l = float(v)
            break
    return t_l, t_u


def region_stats(p, y, t_l, t_u):
    acc, rej = p >= t_u, p < t_l
    gray = ~acc & ~rej
    n = len(p)
    return {
        "n_pairs": int(n),
        "auto_accept": int(acc.sum()), "auto_reject": int(rej.sum()),
        "gray": int(gray.sum()),
        "L_B2": round(float(gray.mean()), 4),
        "false_match_rate_accepted": round(float(1 - y[acc].mean()), 4) if acc.any() else None,
        "missed_match_rate_rejected": round(float(y[rej].mean()), 4) if rej.any() else None,
        "gray_positive_share": round(float(y[gray].mean()), 4) if gray.any() else None,
    }


def main():
    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    cand = pd.read_csv(os.path.join(DERIVED, "candidates.csv"))
    cand = cand[cand["split"] == "cal"].reset_index(drop=True)   # test dropped here
    assert set(cand["split"]) == {"cal"}

    anchors = sorted(cand["anchor_rid"].unique().tolist())
    rnd = random.Random(SEED)
    rnd.shuffle(anchors)
    chosen = set(anchors[:N_CAL_ANCHORS])
    cal = cand[cand["anchor_rid"].isin(chosen)].reset_index(drop=True)

    need = set(cal["anchor_rid"]) | set(cal["cand_rid"])
    recs = {}
    with open(os.path.join(DERIVED, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["rid"] in need:
                recs[r["rid"]] = r
    assert all(recs[i]["partner_id"] == "" for i in need), "partner_id must be blank"

    cal["b2"] = [b2(recs[a], recs[c]) for a, c in zip(cal["anchor_rid"], cal["cand_rid"])]
    y = cal["label"].to_numpy()
    x = cal[["b2"]].to_numpy()

    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(x, y)
    b0, b1 = float(lr.intercept_[0]), float(lr.coef_[0][0])
    if b1 <= 0:
        sys.exit(f"non-monotone calibration (b1={b1}); stop and log a deviation")
    p = 1 / (1 + np.exp(-(b0 + b1 * cal["b2"].to_numpy())))
    cal["p"] = p

    t_l, t_u = thresholds(p, y, ALPHA, BETA)
    no_gray = t_l >= t_u
    stats = region_stats(p, y, t_l, t_u)

    # score-level view of the calibration curve (for the record and the reviewers)
    sc = cal.groupby(cal["b2"].round(6)).agg(n=("label", "size"), pos=("label", "sum"),
                                            p=("p", "first")).reset_index()
    top = sc.sort_values("n", ascending=False).head(15)

    out = {
        "task": "gray-zone-triage Task 3",
        "seed": SEED, "alpha": ALPHA, "beta": BETA,
        "calibration_sample": {
            "anchor_records": len(chosen), "anchor_persons": int(cal["anchor_person"].nunique()),
            "pairs": int(len(cal)), "positives": int(y.sum()),
            "first_anchor_rids": anchors[:5],
        },
        "logistic": {"b0": b0, "b1": b1,
                     "mean_p": round(float(p.mean()), 4), "base_rate": round(float(y.mean()), 4)},
        "thresholds": {"T_L": t_l, "T_U": (None if math.isinf(t_u) else t_u),
                       "T_U_is_infinite": math.isinf(t_u),
                       "T_L_b2_score": float(cal.loc[cal["p"] >= t_l, "b2"].min()),
                       "T_U_b2_score": (None if math.isinf(t_u)
                                        else float(cal.loc[cal["p"] >= t_u, "b2"].min()))},
        "criterion_5_no_gray_zone": bool(no_gray),
        "calibration_regions": stats,
        "distinct_b2_scores": int(sc.shape[0]),
        "largest_score_ties": top.to_dict(orient="records"),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__},
        "wall_time_s": round(time.time() - t0, 1),
    }
    json.dump(out, open(os.path.join(RESULTS, "thresholds.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("calibration_sample", "logistic", "thresholds",
                                          "criterion_5_no_gray_zone", "calibration_regions",
                                          "distinct_b2_scores")}, indent=1))
    print("largest score ties (b2, n, pos, p):")
    for r in out["largest_score_ties"]:
        print(f"  {r['b2']:>9.4f} {r['n']:>6} {r['pos']:>6} {r['p']:.4f}")


if __name__ == "__main__":
    main()
