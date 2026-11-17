"""Task 4 (gray-zone-triage plan): draw the test sample and extract gray zones.

Frozen inputs: results/thresholds.json (Task 3, commit 6b0e803). Thresholds are
applied with the SAVED coefficients b0/b1, never a refit.

1. Calibration gray zone: rebuild the Task 3 calibration sample exactly (cal
   anchor records sorted, random.Random(0) shuffle, first 3,000), score with
   frozen b2, take pairs with T_L <= p < T_U. Asserted to match the 329 pairs
   recorded in thresholds.json. Plan: all of them if <= 800, else a seeded 800.
2. Test sample: test anchor records sorted ascending, shuffled with
   random.Random(0); anchors are added in that order, each with all its pairs,
   until the test gray zone holds >= 600 pairs (the crossing anchor is kept
   whole). Gray membership uses B2 score and the frozen thresholds only.
   Labels are NOT read for this step.
3. Outputs (input/data/derived/):
   gray_cal.jsonl   pair_id, a, b            (no label)
   gray_test.jsonl  pair_id, a, b            (no label)
   test_sample.csv  pair_id, anchor_rid, anchor_person, cand_rid, wave_a,
                    wave_b, b2, p, region    (no label; region in accept/reject/gray)
   labels_test.jsonl pair_id, label for every test-sample pair: a mechanical
                    copy for Task 7. Nothing in this script computes on it.
"""
import json
import os
import random
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(PROJ)
DERIVED = os.path.join(PROJ, "input", "data", "derived")
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(REPO, "src"))
from baseline import b2  # noqa: E402  frozen rule

SEED = 0
N_CAL_ANCHORS = 3000
CAL_GRAY_CAP = 800
TEST_GRAY_TARGET = 600


def load_records(need):
    recs = {}
    with open(os.path.join(DERIVED, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["rid"] in need:
                recs[r["rid"]] = r
    assert len(recs) == len(need), "missing records"
    assert all(r["partner_id"] == "" for r in recs.values()), "partner_id must be blank"
    return recs


def seeded_anchor_order(df):
    anchors = sorted(df["anchor_rid"].unique().tolist())
    random.Random(SEED).shuffle(anchors)
    return anchors


def main():
    th = json.load(open(os.path.join(RESULTS, "thresholds.json")))
    b0, b1 = th["logistic"]["b0"], th["logistic"]["b1"]
    t_l, t_u = th["thresholds"]["T_L"], th["thresholds"]["T_U"]
    assert not th["criterion_5_no_gray_zone"] and t_u is not None

    def prob(s):
        return 1 / (1 + np.exp(-(b0 + b1 * np.asarray(s, dtype=float))))

    def region(p):
        return np.where(p >= t_u, "accept", np.where(p < t_l, "reject", "gray"))

    cols = ["pair_id", "anchor_rid", "cand_rid", "anchor_person", "wave_a", "wave_b", "split"]
    cand = pd.read_csv(os.path.join(DERIVED, "candidates.csv"))
    labels = cand[["pair_id", "label"]]                 # held aside, untouched until output
    cand = cand[cols]

    # ---- 1. calibration gray zone (rebuild Task 3 sample) ---------------------
    cal_all = cand[cand["split"] == "cal"]
    chosen = set(seeded_anchor_order(cal_all)[:N_CAL_ANCHORS])
    cal = cal_all[cal_all["anchor_rid"].isin(chosen)].copy()
    recs = load_records(set(cal["anchor_rid"]) | set(cal["cand_rid"]))
    cal["b2"] = [b2(recs[a], recs[c]) for a, c in zip(cal["anchor_rid"], cal["cand_rid"])]
    cal["p"] = prob(cal["b2"])
    cal["region"] = region(cal["p"].to_numpy())
    assert len(cal) == th["calibration_sample"]["pairs"]
    gc = cal[cal["region"] == "gray"]
    assert len(gc) == th["calibration_regions"]["gray"], (len(gc), th["calibration_regions"]["gray"])
    if len(gc) > CAL_GRAY_CAP:
        keep = sorted(random.Random(SEED).sample(sorted(gc["pair_id"]), CAL_GRAY_CAP))
        gc = gc[gc["pair_id"].isin(keep)]

    # ---- 2. test sample (labels not read) -------------------------------------
    test_all = cand[cand["split"] == "test"]
    order = seeded_anchor_order(test_all)
    by_anchor = {a: g for a, g in test_all.groupby("anchor_rid")}
    need_all = set(test_all["anchor_rid"]) | set(test_all["cand_rid"])
    trecs = load_records(need_all)
    taken, n_gray = [], 0
    for a in order:
        g = by_anchor[a].copy()
        g["b2"] = [b2(trecs[x], trecs[c]) for x, c in zip(g["anchor_rid"], g["cand_rid"])]
        g["p"] = prob(g["b2"])
        g["region"] = region(g["p"].to_numpy())
        taken.append(g)
        n_gray += int((g["region"] == "gray").sum())
        if n_gray >= TEST_GRAY_TARGET:
            break
    test = pd.concat(taken, ignore_index=True)
    gt = test[test["region"] == "gray"]
    assert len(gt) >= TEST_GRAY_TARGET

    # score-cutoff equivalence check (monotone calibration)
    for df in (cal, test):
        s, r = df["b2"].to_numpy(), df["region"].to_numpy()
        assert (s[r == "accept"].min() >= th["thresholds"]["T_U_b2_score"] - 1e-9)
        assert (s[r != "accept"].max() < th["thresholds"]["T_U_b2_score"])
        assert (s[r == "reject"].max() < th["thresholds"]["T_L_b2_score"])

    # ---- 3. outputs -------------------------------------------------------------
    def write_gray(df, fn, rr):
        with open(os.path.join(DERIVED, fn), "w", encoding="utf-8") as f:
            for pid, a, c in zip(df["pair_id"], df["anchor_rid"], df["cand_rid"]):
                f.write(json.dumps({"pair_id": int(pid), "a": rr[a], "b": rr[c]},
                                   ensure_ascii=False) + "\n")

    write_gray(gc, "gray_cal.jsonl", recs)
    write_gray(gt, "gray_test.jsonl", trecs)
    test[["pair_id", "anchor_rid", "anchor_person", "cand_rid", "wave_a", "wave_b",
          "b2", "p", "region"]].to_csv(os.path.join(DERIVED, "test_sample.csv"), index=False)
    lt = labels[labels["pair_id"].isin(set(test["pair_id"]))]
    with open(os.path.join(DERIVED, "labels_test.jsonl"), "w") as f:
        for pid, lab in zip(lt["pair_id"], lt["label"]):
            f.write(json.dumps({"pair_id": int(pid), "label": int(lab)}) + "\n")
    assert len(lt) == len(test)

    summary = {
        "task": "gray-zone-triage Task 4",
        "cal_gray_pairs": int(len(gc)), "cal_gray_capped": len(gc) == CAL_GRAY_CAP,
        "test_anchor_records": len(taken),
        "test_anchor_persons": int(test["anchor_person"].nunique()),
        "test_pairs": int(len(test)),
        "test_regions": test["region"].value_counts().to_dict(),
        "test_gray_pairs": int(len(gt)),
        "L_B2_test": round(float((test["region"] == "gray").mean()), 4),
        "test_wave_pairs": test.groupby("wave_a").size().to_dict(),
    }
    json.dump(summary, open(os.path.join(RESULTS, "test_sample_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
