"""Protocol 4: Splink (no labels, no curated IDs) on the within-wave task, plus
the kin-ID leakage check. Reuses pair building and evaluation from
noise_effect.py; writes *_p4 outputs so protocol-3 files stay untouched."""
import json, math, os, sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import splink.comparison_level_library as cll
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import noise_effect as ne
from evidence import lab

KIN_ID_COLS = [7, 8, 9]  # partner / father / mother person-ID same-or-different
NEVER, EVER = "'Ó'", "('G','E','S')"


def records(df, par):
    """One Splink record per labelled row. No curated ID columns."""
    name_of_row = dict(zip(df.id, df.nafn_norm.str.strip()))
    ll = {}
    for pid, r in par.iterrows():
        try:
            ll[pid] = (float(r.lat), float(r.lon))
        except ValueError:
            pass
    nz = lambda v: v.strip().lower() if isinstance(v, str) and v.strip() else None
    L = df[df.person.map(lab)]
    out = pd.DataFrame({
        "unique_id": L.id.astype(int).values,
        "wave": L.heimild.astype(int).values,
        "first_name": L.first_name.map(nz).values,
        "patronym": L.patronym.map(nz).values,
        "birthyear": L.by.values,
        "parish": L.parish.map(nz).values, "county": L.county.map(nz).values, "farm": L.farm.map(nz).values,
        "lat": [ll.get(p, (None, None))[0] for p in L.parish], "lon": [ll.get(p, (None, None))[1] for p in L.parish],
        "partner_name": [nz(name_of_row.get(s, "")) if s else None for s in L.source_partner],
        "marriage": [m.strip() if m.strip() else None for m in L.marriagestatus],
    })
    out["birthyear"] = out.birthyear.astype("Int64")
    return out


def settings():
    by = cl.CustomComparison(output_column_name="birthyear", comparison_levels=[
        cll.NullLevel("birthyear"), cll.ExactMatchLevel("birthyear"),
        cll.CustomLevel("abs(birthyear_l - birthyear_r) <= 1", "within 1"),
        cll.CustomLevel("abs(birthyear_l - birthyear_r) <= 2", "within 2"), cll.ElseLevel()])
    mar = cl.CustomComparison(output_column_name="marriage", comparison_levels=[
        cll.NullLevel("marriage"), cll.ExactMatchLevel("marriage"),
        cll.CustomLevel(f"(marriage_l = {NEVER} AND marriage_r IN {EVER}) OR (marriage_r = {NEVER} AND marriage_l IN {EVER})",
                        "never-married vs ever-married"), cll.ElseLevel()])
    return SettingsCreator(
        link_type="dedupe_only",
        comparisons=[cl.JaroWinklerAtThresholds("first_name", [0.9]), cl.JaroWinklerAtThresholds("patronym", [0.9]),
                     by, cl.ExactMatch("parish"), cl.ExactMatch("county"), cl.ExactMatch("farm"),
                     cl.DistanceInKMAtThresholds("lat", "lon", [1, 10, 50]), cl.ExactMatch("partner_name"), mar],
        blocking_rules_to_generate_predictions=[block_on("wave", "first_name", "patronym")],
        additional_columns_to_retain=["wave"], retain_intermediate_calculation_columns=False)


def train(recs):
    linker = Linker(recs, settings(), DuckDBAPI())
    linker.training.estimate_probability_two_random_records_match(
        [block_on("wave", "first_name", "patronym", "birthyear", "parish")], recall=0.8)
    linker.training.estimate_u_using_random_sampling(max_pairs=1e7, seed=0)
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("wave", "first_name", "patronym"))
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("wave", "birthyear", "parish"))
    return linker


def score_pairs(model, recs, pairs):
    """Score exactly the evaluation pairs: link_only, blocked on a pair id."""
    by_id = recs.set_index("unique_id")
    a = by_id.loc[[int(x) for x, _ in pairs]].reset_index(drop=True)
    b = by_id.loc[[int(y) for _, y in pairs]].reset_index(drop=True)
    for t in (a, b):
        t.insert(0, "unique_id", np.arange(len(pairs)))
        t["pair_id"] = np.arange(len(pairs))
    s = dict(model)
    s["link_type"] = "link_only"
    s["blocking_rules_to_generate_predictions"] = [{"blocking_rule": "l.pair_id = r.pair_id", "sql_dialect": "duckdb"}]
    s["additional_columns_to_retain"] = ["pair_id"]
    lk = Linker([a, b], s, DuckDBAPI(), input_table_aliases=["a", "b"])
    pred = lk.inference.predict().as_pandas_dataframe()
    mw = dict(zip(pred.pair_id_l, pred.match_weight))
    missing = sum(i not in mw for i in range(len(pairs)))
    assert missing == 0, f"{missing} pairs not scored"
    return np.array([mw[i] for i in range(len(pairs))])


def extra(B):
    recs = records(B["df"], B["par"])
    print(f"training Splink on {len(recs)} labelled rows (no labels used)")
    linker = train(recs)
    model = linker.misc.save_model_to_json(os.path.join(HERE, "splink_within_wave_model.json"), overwrite=True)
    S = {"Splink": score_pairs(model, recs, B["pairs"])}
    # leakage check: drop curated kin-ID features
    X, y, g = B["X"], B["y_ice"], B["groups"]
    keep = [j for j in range(X.shape[1]) if j not in KIN_ID_COLS]
    S["B2_noid"] = B["S"]["B2"] - 2.0 * (X[:, 7] == 1)
    for name, mk in [("LR_noid", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                     ("GB_noid", lambda: HistGradientBoostingClassifier(random_state=ne.SEED))]:
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            oof[te] = mk().fit(X[tr][:, keep], y[tr]).predict_proba(X[te][:, keep])[:, 1]
        S[name] = oof
    return S


def analysis():
    """Paired leads and linked shares, as in noise_effect_extra.py, for the p4 methods."""
    d = np.load(os.path.join(HERE, "noise_effect_scores_p4.npz"), allow_pickle=True)
    idx, yI, yA, st, w, y_ice, y_rule = (d[k] for k in ["idx", "yI", "yA", "st", "w", "y_ice", "y_rule"])
    S = {k[2:]: d[k] for k in d.files if k.startswith("S_")}
    rng = np.random.default_rng(0)
    strata = [np.where(st == g_)[0] for g_ in np.unique(st)]
    boots = [np.concatenate([rng.choice(ix, len(ix)) for ix in strata]) for _ in range(1000)]
    over = (st != "neg") & (yA == 0)
    out = {"lead_primary": {}, "linked_share_of_over_merges": {}}
    for m, ref in [("GB_noid", "B2_noid"), ("LR_noid", "B2_noid"), ("Splink", "B2_noid"), ("Splink", "GB_noisy"), ("Splink", "GB_noid")]:
        a, b = S[m][idx], S[ref][idx]
        res = {}
        for nm, y in [("ice", yI), ("clean", yA)]:
            vals = [roc_auc_score(y[t], a[t], sample_weight=w[t]) - roc_auc_score(y[t], b[t], sample_weight=w[t]) for t in boots]
            res[nm] = {"est": roc_auc_score(y, a, sample_weight=w) - roc_auc_score(y, b, sample_weight=w),
                       "ci": np.percentile(vals, [2.5, 97.5]).tolist()}
        out["lead_primary"][f"{m} - {ref}"] = res
    def shares(s, thr):
        def share(tk):
            o = over[tk]
            return float((w[tk][o] * (s[tk][o] >= thr)).sum() / w[tk][o].sum())
        return {"est": share(np.arange(len(s))), "ci": np.percentile([share(t) for t in boots], [2.5, 97.5]).tolist(),
                "true_dups_linked": float((w[yA == 1] * (s[yA == 1] >= thr)).sum() / w[yA == 1].sum()), "threshold": float(thr)}
    for m, s_all in S.items():
        s = s_all[idx]
        p, r, t = precision_recall_curve(yI, s, sample_weight=w)
        f1 = 2 * p * r / np.maximum(p + r, 1e-12)
        out["linked_share_of_over_merges"][m] = shares(s, t[np.argmax(f1[:-1])])
    out["linked_share_of_over_merges"]["Splink@p0.5"] = shares(S["Splink"][idx], 0.0)  # match weight 0 = p 0.5
    json.dump(out, open(os.path.join(HERE, "splink_p4_extra.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ne.main(extra=extra, tag="_p4")
    analysis()
