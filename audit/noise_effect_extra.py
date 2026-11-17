"""NOT pre-registered. (1) Paired bootstrap of the learners' AUC lead over B2
under each label condition. (2) Share of audited over-merges each method links
at the threshold that maximises F1 under ICE-ID labels."""
import json, os
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "noise_effect_scores.npz"), allow_pickle=True)
idx, yI, yA, st, w, y_ice, y_rule = (d[k] for k in ["idx", "yI", "yA", "st", "w", "y_ice", "y_rule"])
S = {k[2:]: d[k] for k in d.files if k.startswith("S_")}
rng = np.random.default_rng(0)
strata = {g: np.where(st == g)[0] for g in np.unique(st)}
boots = [np.concatenate([rng.choice(ix, len(ix)) for ix in strata.values()]) for _ in range(1000)]
over = (st != "neg") & (yA == 0)
out = {"lead_over_B2_primary": {}, "linked_share_of_over_merges": {}}
for m in ["GB_noisy", "LR_noisy", "name_age"]:
    a, b = S[m][idx], S["B2"][idx]
    res = {}
    for nm, y in [("ice", yI), ("clean", yA)]:
        vals = [roc_auc_score(y[t], a[t], sample_weight=w[t]) - roc_auc_score(y[t], b[t], sample_weight=w[t]) for t in boots]
        res[nm] = {"est": roc_auc_score(y, a, sample_weight=w) - roc_auc_score(y, b, sample_weight=w), "ci": np.percentile(vals, [2.5, 97.5]).tolist()}
    out["lead_over_B2_primary"][m] = res
for m, s_all in S.items():
    s = s_all[idx]
    p, r, t = precision_recall_curve(yI, s, sample_weight=w)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12); thr = t[np.argmax(f1[:-1])]
    def share(tk):
        o = over[tk]; return float((w[tk][o] * (s[tk][o] >= thr)).sum() / w[tk][o].sum())
    vals = [share(tk) for tk in boots]
    out["linked_share_of_over_merges"][m] = {"est": share(np.arange(len(s))), "ci": np.percentile(vals, [2.5, 97.5]).tolist(),
                                             "true_dups_linked": float((w[yA == 1] * (s[yA == 1] >= thr)).sum() / w[yA == 1].sum())}
# secondary: lead over B2 on all pairs (rule-cleaned), plain bootstrap
n = len(y_ice); sec = {}
bt = [rng.integers(0, n, n) for _ in range(200)]
for m in ["GB_noisy", "LR_noisy"]:
    a, b = S[m], S["B2"]; res = {}
    for nm, y in [("ice", y_ice), ("clean", y_rule)]:
        vals = [roc_auc_score(y[t], a[t]) - roc_auc_score(y[t], b[t]) for t in bt]
        res[nm] = {"est": roc_auc_score(y, a) - roc_auc_score(y, b), "ci": np.percentile(vals, [2.5, 97.5]).tolist()}
    sec[m] = res
out["lead_over_B2_secondary"] = sec
json.dump(out, open(os.path.join(HERE, "noise_effect_extra.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
