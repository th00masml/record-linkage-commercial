"""Task 5 (gray-zone-triage plan): Fellegi-Sunter second stage (Splink).

Unsupervised: no labels are read in this script.

Model (plan Task 5, Amendment 1):
- Comparisons on the B2 fields except partner (blanked, Amendment 1):
  birth year (null / exact / |d|<=1 / |d|<=2 / else), parish, county, farm
  (exact match, null-aware), distance in km between parish centroids
  (<=0.001 i.e. same place / <10 / <50 / else).
- Data: every record that occurs in a calibration-split candidate pair
  (input/data/derived/candidates.csv, split == "cal"); records.jsonl has
  partner_id blanked.
- u: random sampling (max 1e6 pairs, seed 0).
- m: EM, two passes (Splink cannot estimate m for a field used in the EM
  blocking rule):
    pass 1 blocks on first, patronym, sex, adjacent wave, same parish
           -> estimates birth year (and farm, county where identifiable);
    pass 2 blocks exactly like the candidates: first, patronym, sex, adjacent
           wave, |birth year diff| <= 2 -> estimates parish, county, farm,
           distance.
- Scoring: Splink's match weight uses the prior LAMBDA0 = 1e-4 set in the
  settings; the Bayes-factor sum is BF = match_weight - log2(LAMBDA0 / (1 - LAMBDA0)).
- In-block prior (decision recorded before any outcome, see log): lambda_b is
  the EM fixed point on the calibration candidate pairs themselves,
  lambda_b = mean_i sigmoid(logit(lambda_b) + BF_i * ln 2), solved by
  iteration. No labels are used. The posterior "among candidates" is
  p_FS = sigmoid(logit(lambda_b) + BF * ln 2); D_FS = 1 if p_FS >= 0.5.
- Output: results/fs_decisions.jsonl for every calibration and test gray-zone
  pair (input/data/derived/gray_{cal,test}.jsonl), and results/fs_model.json.
"""
import json
import logging
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import splink
import splink.comparison_level_library as cll
import splink.comparison_library as cl
from splink import Linker, SettingsCreator
from splink.backends.duckdb import DuckDBAPI

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
DERIVED = os.path.join(PROJ, "input", "data", "derived")
RESULTS = os.path.join(HERE, "results")

SEED = 0
LAMBDA0 = 1e-4
U_MAX_PAIRS = 1e6
BLOCK_CAND = ("l.first = r.first and l.patronym = r.patronym and l.sex = r.sex "
              "and abs(l.wave_idx - r.wave_idx) = 1 "
              "and abs(l.birthyear - r.birthyear) <= 2")
BLOCK_PARISH = ("l.first = r.first and l.patronym = r.patronym and l.sex = r.sex "
                "and abs(l.wave_idx - r.wave_idx) = 1 and l.parish = r.parish")


def to_row(r, widx):
    return {
        "unique_id": int(r["rid"]),
        "first": r["first"], "patronym": r["patronym"], "sex": r["sex"],
        "birthyear": int(r["birthyear"]), "wave_idx": widx[int(r["wave"])],
        "parish": r["parish"] or None, "county": r["county"] or None,
        "farm_id": r["farm_id"] or None,
        "lat": float(r["lat"]) if r["lat"] else None,
        "lon": float(r["lon"]) if r["lon"] else None,
    }


def settings():
    by = cl.CustomComparison(
        comparison_levels=[cll.NullLevel("birthyear"), cll.ExactMatchLevel("birthyear"),
                           cll.AbsoluteDifferenceLevel("birthyear", 1),
                           cll.AbsoluteDifferenceLevel("birthyear", 2), cll.ElseLevel()],
        output_column_name="birthyear")
    return SettingsCreator(
        link_type="dedupe_only",
        comparisons=[by, cl.ExactMatch("parish"), cl.ExactMatch("county"),
                     cl.ExactMatch("farm_id"),
                     cl.DistanceInKMAtThresholds("lat", "lon", [0.001, 10, 50])],
        blocking_rules_to_generate_predictions=[BLOCK_CAND],
        probability_two_random_records_match=LAMBDA0,
        retain_intermediate_calculation_columns=True,
    )


def logit2(p):
    return math.log2(p / (1 - p))


def main():
    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    cand = pd.read_csv(os.path.join(DERIVED, "candidates.csv"),
                       usecols=["pair_id", "anchor_rid", "cand_rid", "split"])
    cal = cand[cand["split"] == "cal"][["pair_id", "anchor_rid", "cand_rid"]]

    gray = {s: [json.loads(l) for l in open(os.path.join(DERIVED, f"gray_{s}.jsonl"))]
            for s in ("cal", "test")}
    assert all("label" not in p for s in gray for p in gray[s])

    need = set(cal["anchor_rid"]) | set(cal["cand_rid"])
    recs, waves = {}, set()
    with open(os.path.join(DERIVED, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            waves.add(int(r["wave"]))
            if r["rid"] in need:
                recs[r["rid"]] = r
    widx = {w: i for i, w in enumerate(sorted(waves))}
    assert len(widx) == 16
    assert all(r["partner_id"] == "" for r in recs.values())
    df = pd.DataFrame([to_row(r, widx) for r in recs.values()])

    db = DuckDBAPI()
    linker = Linker(db.register(df), settings(), log_level=logging.WARNING)
    linker.training.estimate_u_using_random_sampling(max_pairs=U_MAX_PAIRS, seed=SEED)
    linker.training.estimate_parameters_using_expectation_maximisation(BLOCK_PARISH)
    linker.training.estimate_parameters_using_expectation_maximisation(BLOCK_CAND)

    model_path = os.path.join(RESULTS, "fs_model.json")
    linker.misc.save_model_to_json(model_path, overwrite=True)
    model = json.load(open(model_path))
    untrained = []
    for c in model["comparisons"]:
        for lv in c["comparison_levels"]:
            if lv.get("is_null_level"):
                continue
            if lv.get("m_probability") is None or lv.get("u_probability") is None:
                untrained.append(f'{c["output_column_name"]}: {lv["label_for_charts"]}')

    # ---- BF sums on the calibration candidate pairs -> in-block prior ----------
    pred = linker.inference.predict().as_pandas_dataframe()
    a = pred[["unique_id_l", "unique_id_r"]].min(axis=1)
    b = pred[["unique_id_l", "unique_id_r"]].max(axis=1)
    pred = pred.assign(lo=a.astype("int64"), hi=b.astype("int64"))
    cal = cal.assign(lo=cal[["anchor_rid", "cand_rid"]].min(axis=1),
                     hi=cal[["anchor_rid", "cand_rid"]].max(axis=1))
    m = cal.merge(pred[["lo", "hi", "match_weight"]], on=["lo", "hi"], how="left")
    assert m["match_weight"].notna().all(), "some calibration candidate pairs not scored"
    bf_cal = (m["match_weight"] - logit2(LAMBDA0)).to_numpy()

    lam = 0.5
    for it in range(1000):
        z = math.log(lam / (1 - lam)) + bf_cal * math.log(2)
        new = float(np.mean(1 / (1 + np.exp(-z))))
        if abs(new - lam) < 1e-10:
            lam = new
            break
        lam = new
    lam_b = lam

    # ---- score gray-zone pairs ---------------------------------------------------
    out_rows = []
    for s in ("cal", "test"):
        for p in gray[s]:
            res = linker.inference.score_pair(to_row(p["a"], widx),
                                              to_row(p["b"], widx)).as_pandas_dataframe()
            mw = float(res["match_weight"].iloc[0])
            bf = mw - logit2(LAMBDA0)
            z = math.log(lam_b / (1 - lam_b)) + bf * math.log(2)
            p_fs = 1 / (1 + math.exp(-z)) if z > -700 else 0.0
            out_rows.append({"split": s, "pair_id": p["pair_id"], "match_weight": mw,
                             "bf_log2": bf, "p_fs": p_fs, "D_fs": int(p_fs >= 0.5)})
    with open(os.path.join(RESULTS, "fs_decisions.jsonl"), "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    # cross-check: score_pair agrees with predict() on calibration gray pairs
    cg = {r["pair_id"]: r["match_weight"] for r in out_rows if r["split"] == "cal"}
    mm = m.set_index("pair_id")["match_weight"]
    diffs = [abs(cg[k] - mm[k]) for k in cg]
    assert max(diffs) < 1e-6, f"score_pair vs predict mismatch {max(diffs)}"

    od = pd.DataFrame(out_rows)
    summary = {
        "task": "gray-zone-triage Task 5",
        "splink": splink.__version__, "seed": SEED, "lambda0": LAMBDA0,
        "records_in_model": int(len(df)), "cal_candidate_pairs_scored": int(len(m)),
        "lambda_inblock": lam_b, "lambda_iterations": it + 1,
        "untrained_levels": untrained,
        "gray_scored": od.groupby("split").size().to_dict(),
        "D_fs_share": od.groupby("split")["D_fs"].mean().round(4).to_dict(),
        "wall_time_s": round(time.time() - t0, 1),
    }
    json.dump(summary, open(os.path.join(RESULTS, "fs_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
