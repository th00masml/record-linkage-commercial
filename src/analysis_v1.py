"""Post-hoc analysis of the preregistered v1 set (200 pairs, cached LLM scores).
Not part of the preregistered test; the v1 verdict is unchanged by it.

1. Leak diagnostics: how much class signal is in fields that should carry none.
2. B2 without the partner term (v1 `partner` is a gold-derived person id).
3. Fellegi-Sunter fitted on v2 dev (disjoint names), applied to v1.
4. LLM as a feature: cross-fitted logistic stacking of [B2, logit(LLM)] vs B2 alone.

Writes outputs/analysis_v1.json.
"""
import json
import os

import numpy as np

from baseline import b2
from common import paired_delta_ci, roc_auc
from linkage import FellegiSunter, comparison_matrix, fit_logistic
from prepare_pairs_v2 import OUTDIR as PAIRDIR

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
LLMS = {"M_comm_sonnet5": "claude-sonnet-5", "M_local_qwen14b": "qwen2.5:14b"}


def logit(p, eps=0.005):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def zscore(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def crossfit(X, y, k=10, reps=20, l2=1.0):
    """Out-of-fold scores averaged over `reps` random stratified k-fold splits.

    Only the direction of the fitted slope vector is used (no intercept, unit norm):
    per-fold intercepts and scales would otherwise shuffle the pooled ranking and
    bias AUC downward. With one feature this reproduces that feature's ranking."""
    pred = np.zeros(len(y))
    for r in range(reps):
        rng = np.random.default_rng(r)
        fold = np.empty(len(y), int)
        for c in (0, 1):
            idx = np.flatnonzero(y == c)
            fold[idx] = rng.permutation(len(idx)) % k
        for f in range(k):
            tr, te = fold != f, fold == f
            w = fit_logistic(X[tr], y[tr], l2=l2)[1:]
            pred[te] += X[te] @ (w / np.linalg.norm(w))
    return pred / reps


def main():
    pairs = [json.loads(l) for l in open(os.path.join(ROOT, "data/derived/pairs.jsonl"), encoding="utf-8")]
    y = np.array([p["label"] for p in pairs])
    res = {}

    gap = [-abs(int(p["a"]["wave"]) - int(p["b"]["wave"])) for p in pairs]
    order = [int(int(p["a"]["wave"]) < int(p["b"]["wave"])) for p in pairs]
    namediff = [int(p["a"]["name"] == p["b"]["name"]) for p in pairs]
    res["leaks"] = {
        "auc_census_gap_only": roc_auc(y, gap),
        "auc_wave_order_only": roc_auc(y, order),
        "auc_full_name_identical_only": roc_auc(y, namediff),
        "neg_with_A_later_than_B": int(sum(1 - o for o, l in zip(order, y) if l == 0)),
        "pos_with_A_later_than_B": int(sum(1 - o for o, l in zip(order, y) if l == 1)),
        "pos_full_name_differs": int(sum(1 - n for n, l in zip(namediff, y) if l == 1)),
        "neg_full_name_differs": int(sum(1 - n for n, l in zip(namediff, y) if l == 0)),
        "neg_first_letter_counts": dict(sorted(
            {c: sum(1 for p in pairs if p["label"] == 0 and p["a"]["first"][0] == c)
             for c in {p["a"]["first"][0] for p in pairs if p["label"] == 0}}.items())),
    }

    s_b2 = np.array([b2(p["a"], p["b"]) for p in pairs])
    no_partner = s_b2 - np.array([2.0 if p["a"]["partner_id"] and p["a"]["partner_id"] == p["b"]["partner_id"]
                                  else 0.0 for p in pairs])
    res["B2"] = roc_auc(y, s_b2)
    res["B2_without_gold_partner"] = roc_auc(y, no_partner)

    # FS fitted on v2 dev (no name overlap issue: v1 names are a different sample,
    # and FS uses no name field). v1 records lack partner_name -> treated as missing.
    dev = [json.loads(l) for l in open(os.path.join(PAIRDIR, "dev_adjacent.jsonl"), encoding="utf-8")]
    Gd, yd = comparison_matrix(dev), np.array([p["label"] for p in dev])
    G = comparison_matrix(pairs)
    res["FS_em_from_v2_dev"] = roc_auc(y, FellegiSunter.fit_em(Gd).weights(G))
    res["FS_sup_from_v2_dev"] = roc_auc(y, FellegiSunter.fit_supervised(Gd, yd).weights(G))

    base = crossfit(zscore(no_partner[:, None]), y)
    res["stacking"] = {"B2_nopartner_crossfit_auc": roc_auc(y, base)}
    for key, label in LLMS.items():
        llm = np.array(json.load(open(os.path.join(OUT, f"scores_{key}.json")))["scores"])
        X = zscore(np.column_stack([no_partner, logit(llm)]))
        hyb = crossfit(X, y)
        obs, lo, hi, p = paired_delta_ci(y, base, hyb)
        res["stacking"][label] = {"hybrid_auc": roc_auc(y, hyb), "delta_vs_B2": obs,
                                  "delta_ci95": [lo, hi], "p_delta_le_0": p}
    json.dump(res, open(os.path.join(OUT, "analysis_v1.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
