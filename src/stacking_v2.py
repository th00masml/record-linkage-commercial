"""LLM as a feature on top of the rule, on v2 llm_adjacent (post-hoc).

Same cross-fitted stacking as analysis_v1.py: [B2, logit(LLM)] vs B2 alone, slope
direction only. Uses every LLM run found in outputs/v2_llm_adjacent__*/.
Writes outputs/v2/stacking_llm_adjacent.json.
"""
import glob
import json
import os

import numpy as np

from analysis_v1 import crossfit, logit, zscore
from common import paired_delta_ci, roc_auc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "outputs")


def main():
    b2 = json.load(open(os.path.join(OUT, "v2", "scores_llm_adjacent_B2.json")))
    y = np.array(b2["labels"])
    s_b2 = np.array(b2["scores"])
    base = crossfit(zscore(s_b2[:, None]), y)
    res = {"B2_crossfit_auc": roc_auc(y, base), "runs": {}}
    for path in sorted(glob.glob(os.path.join(OUT, "v2_llm_adjacent__*", "scores_*.json"))):
        r = json.load(open(path))
        assert r["pid"] == b2["pid"], path
        tag = os.path.basename(os.path.dirname(path)).split("__", 1)[1]
        hyb = crossfit(zscore(np.column_stack([s_b2, logit(r["scores"])])), y)
        obs, lo, hi, p = paired_delta_ci(y, base, hyb)
        res["runs"][tag] = {"llm_auc": r["auc"], "hybrid_auc": roc_auc(y, hyb),
                            "delta_vs_B2": obs, "delta_ci95": [lo, hi], "p_delta_le_0": p}
    json.dump(res, open(os.path.join(OUT, "v2", "stacking_llm_adjacent.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
