"""Deterministic and probabilistic baselines on the v2 pair sets.

  B1        birth-year agreement only (v1 definition)
  B2        the v1 hand-built rule with its weights FROZEN as preregistered; the
            partner term compares partner NAMES (v2 has no gold partner id)
  FS_em     Fellegi-Sunter, m/u by unsupervised EM on dev_adjacent (labels unused)
  FS_sup    Fellegi-Sunter, m/u counted from dev_adjacent labels
  LR        L2 logistic regression on one-hot comparison levels, fitted on dev

Nothing is fitted on a test set. Writes outputs/v2/scores_<set>_<method>.json and
outputs/v2/fitted_models.json. Splink is run separately (splink_baseline.py).
"""
import json
import os

import numpy as np

from baseline import b1, hav
from common import roc_auc
from linkage import (FellegiSunter, average_precision, bootstrap_ci, comparison_matrix,
                     fit_logistic, one_hot, predict_logistic)
from prepare_pairs_v2 import OUTDIR as PAIRDIR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "outputs/v2")
TEST_SETS = ["test_adjacent", "test_longgap", "test_natural", "llm_adjacent"]


def b2_v2(a, b):
    """v1 B2 weights unchanged; partner compared by name instead of gold id."""
    s = 0.0
    dy = abs(a["birthyear"] - b["birthyear"])
    s += 2.0 if dy == 0 else (1.0 if dy <= 1 else 0.0)
    if a["parish"] and a["parish"] == b["parish"]:
        s += 1.5
    if a["county"] and a["county"] == b["county"]:
        s += 0.5
    if a["farm_id"] and a["farm_id"] == b["farm_id"]:
        s += 1.0
    if a.get("partner_name") and a["partner_name"] == b.get("partner_name"):
        s += 2.0
    km = hav(a, b)
    if km is not None:
        s += max(0.0, 1.0 - km / 100.0)
    return s


def load(name):
    with open(os.path.join(PAIRDIR, f"{name}.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def record(set_name, method, pairs, scores):
    labels = [p["label"] for p in pairs]
    scores = [float(s) for s in scores]
    res = {"method": method, "set": set_name, "auc": roc_auc(labels, scores),
           "ci95": bootstrap_ci(roc_auc, labels, scores),
           "ap": average_precision(labels, scores),
           "ap_ci95": bootstrap_ci(average_precision, labels, scores),
           "pid": [p["pid"] for p in pairs], "labels": labels, "scores": scores}
    json.dump(res, open(os.path.join(OUT, f"scores_{set_name}_{method}.json"), "w"))
    print(f"{set_name:14} {method:7} AUC={res['auc']:.4f} "
          f"[{res['ci95'][0]:.4f},{res['ci95'][1]:.4f}]  AP={res['ap']:.4f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = load("dev_adjacent")
    G_dev = comparison_matrix(dev)
    y_dev = np.array([p["label"] for p in dev])
    fs_em = FellegiSunter.fit_em(G_dev)
    fs_sup = FellegiSunter.fit_supervised(G_dev, y_dev)
    lr = fit_logistic(one_hot(G_dev), y_dev)
    json.dump({"FS_em": fs_em.to_dict(), "FS_sup": fs_sup.to_dict(),
               "LR_coef": lr.round(5).tolist(), "fitted_on": "dev_adjacent"},
              open(os.path.join(OUT, "fitted_models.json"), "w"), indent=1)
    print(f"dev_adjacent: {len(dev)} pairs; EM lambda={fs_em.lam:.3f} (true {y_dev.mean():.3f})")

    for name in TEST_SETS:
        pairs = load(name)
        G = comparison_matrix(pairs)
        record(name, "B1", pairs, [b1(p["a"], p["b"]) for p in pairs])
        record(name, "B2", pairs, [b2_v2(p["a"], p["b"]) for p in pairs])
        record(name, "FS_em", pairs, fs_em.weights(G))
        record(name, "FS_sup", pairs, fs_sup.weights(G))
        record(name, "LR", pairs, predict_logistic(lr, one_hot(G)))


if __name__ == "__main__":
    main()
