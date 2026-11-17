"""Task 7 (gray-zone-triage plan): primary, secondary and sensitivity analysis.

    python analyze.py            # real run: needs results/raw_qwen_gray.jsonl
    python analyze.py --dry-run  # code check only: the calibration sample plays
                                 # "test", a random decision plays the LLM; test
                                 # labels are NOT read; nothing is written to
                                 # results/primary.json

Pre-registered definitions (plan: Hypothesis, Falsification Criteria, Frozen
parameters, Amendment 1):
- Prior p = sigmoid(b0 + b1 * B2) with the SAVED coefficients; regions:
  p >= T_U accept, p < T_L reject, otherwise gray.
- LLM decision D per gray pair: mean of the parsed scores of the 5 samples
  >= 0.5 -> 1, else 0; no parsed sample -> no D, the pair stays gray.
- Second-stage accuracy on the CALIBRATION gray zone, Laplace-smoothed:
  sens = (TP + 1) / (P + 2), spec = (TN + 1) / (N + 2).
- Posterior odds = prior odds x LR, LR = sens / (1 - spec) if D = 1 else
  (1 - sens) / spec; same thresholds on the posterior.
- Review load L = share of test pairs left gray; R = 1 - L_hyb / L_B2.
- Realized false-match rate = share of non-matches among ALL auto-accepted
  test pairs (B2 accept plus posterior accept); realized missed-match rate =
  share of matches among ALL auto-rejected test pairs.
- 95% CIs: percentile bootstrap, 1,000 resamples of test anchor PERSONS
  (Amendment 1, item 2), seed 0. Second-stage accuracy is held fixed.
- Criteria: (1) R < 0.20, (2) CI lower bound of R <= 0, (3) false-match > 0.05,
  (4) missed-match > 0.05, (5) no gray zone -> KILLED. Otherwise POSITIVE if
  R_LLM - R_FS has a CI lower bound > 0, else MIXED.
Sensitivity (reported, not decisive; Amendment 1, item 3): CI clustered on
name blocks; metrics without anchors touching duplicate (person, wave) keys;
anchor-weighted L. Descriptive (item 4): end-to-end misses, per-wave-pair L
and error rates. Exploratory (log, Task 5): H2 with the D_FS threshold chosen
on the calibration gray zone by maximal Youden J.
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(PROJ)
DERIVED = os.path.join(PROJ, "input", "data", "derived")
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(REPO, "src"))

SEED = 0
N_BOOT = 1000
R_MIN = 0.20
ALPHA = 0.05
BETA = 0.05
SEEDS = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------- helpers ----
def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def accuracy(dec, lab):
    """dec, lab: dict pair_id -> 0/1 (dec may lack pairs with no D)."""
    ids = [i for i in dec if i in lab]
    tp = sum(1 for i in ids if dec[i] == 1 and lab[i] == 1)
    fn = sum(1 for i in ids if dec[i] == 0 and lab[i] == 1)
    tn = sum(1 for i in ids if dec[i] == 0 and lab[i] == 0)
    fp = sum(1 for i in ids if dec[i] == 1 and lab[i] == 0)
    P, N = tp + fn, tn + fp
    sens, spec = (tp + 1) / (P + 2), (tn + 1) / (N + 2)
    return {"TP": tp, "FN": fn, "TN": tn, "FP": fp, "P": P, "N": N,
            "sens": sens, "spec": spec,
            "sens_raw_ci95": wilson(tp, P), "spec_raw_ci95": wilson(tn, N),
            "LR_pos": sens / (1 - spec), "LR_neg": (1 - sens) / spec}


def posterior_regions(df, dec, acc, t_l, t_u):
    """df: test_sample rows with p, region. Returns final region array."""
    reg = df["region"].to_numpy().copy()
    p = df["p"].to_numpy()
    for k, (pid, r) in enumerate(zip(df["pair_id"], reg)):
        if r != "gray" or pid not in dec:
            continue
        lr = acc["LR_pos"] if dec[pid] == 1 else acc["LR_neg"]
        odds = p[k] / (1 - p[k]) * lr
        q = odds / (1 + odds)
        reg[k] = "accept" if q >= t_u else ("reject" if q < t_l else "gray")
    return reg


def metrics(reg0, reg1, y):
    L0, L1 = float(np.mean(reg0 == "gray")), float(np.mean(reg1 == "gray"))
    acc1, rej1 = reg1 == "accept", reg1 == "reject"
    return {"L_B2": L0, "L_hyb": L1, "R": (1 - L1 / L0) if L0 > 0 else None,
            "delta_pp": 100 * (L1 - L0),
            "false_match_accepted": float(1 - y[acc1].mean()) if acc1.any() else None,
            "missed_match_rejected": float(y[rej1].mean()) if rej1.any() else None,
            "n_accept": int(acc1.sum()), "n_reject": int(rej1.sum()),
            "n_gray": int((reg1 == "gray").sum())}


def cluster_boot(df, reg0, regs, y, cluster_col, n_boot=N_BOOT, seed=SEED):
    """Percentile CIs of R per method and of R_a - R_b, resampling clusters."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(df[cluster_col])
    idx_by = pd.Series(np.arange(len(df))).groupby(codes).apply(np.array).to_list()
    g0 = (reg0 == "gray").astype(float)
    gs = {k: (v == "gray").astype(float) for k, v in regs.items()}
    out = {k: [] for k in regs}
    out["LLM_minus_FS"] = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        ix = np.concatenate([idx_by[i] for i in pick])
        l0 = g0[ix].mean()
        rr = {k: (1 - gs[k][ix].mean() / l0) if l0 > 0 else np.nan for k in regs}
        for k in regs:
            out[k].append(rr[k])
        if "LLM" in rr and "FS" in rr:
            out["LLM_minus_FS"].append(rr["LLM"] - rr["FS"])
    ci = {}
    for k, v in out.items():
        v = np.array(v, dtype=float)
        v = v[~np.isnan(v)]
        ci[k] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    return ci, int(len(uniq))


def llm_decisions(raw, split):
    rows = [r for r in raw if r["split"] == split]
    by = {}
    for r in rows:
        by.setdefault(r["pair_id"], {})[r["seed"]] = r["score"]
    dec, none, unparsed = {}, 0, 0
    for pid, s in by.items():
        assert sorted(s) == SEEDS, f"pair {pid} ({split}) lacks seeds: {sorted(s)}"
        vals = [v for v in s.values() if v is not None]
        unparsed += len(SEEDS) - len(vals)
        if not vals:
            none += 1
            continue
        dec[pid] = int(np.mean(vals) >= 0.5)
    return dec, {"pairs": len(by), "no_decision": none,
                 "unparsed_samples": unparsed, "samples": len(rows),
                 "D1_share": round(float(np.mean(list(dec.values()))), 4) if dec else None}


def git_order_ok():
    """Verification: thresholds.json commit precedes gray_test, which precedes raw_qwen."""
    def first_commit(path):
        out = subprocess.run(["git", "-C", REPO, "log", "--diff-filter=A", "--format=%H",
                              "--", path], capture_output=True, text=True).stdout.split()
        return out[-1] if out else None

    def before(a, b):
        return subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", a, b]).returncode == 0

    rel = "hybrid-census-linkage/"
    c = [first_commit(rel + p) for p in ("output/data-analysis/results/thresholds.json",
                                          "input/data/derived/gray_test.jsonl",
                                          "output/data-analysis/results/raw_qwen_gray.jsonl")]
    if None in c:
        return {"commits": c, "ordered": None}
    return {"commits": c, "ordered": before(c[0], c[1]) and before(c[1], c[2]) and c[0] != c[1] != c[2]}


# ------------------------------------------------------------------- data ----
def load_cal_sample(th):
    """Rebuild the Task 3 calibration sample with region (dry run uses it as test)."""
    from baseline import b2
    cand = pd.read_csv(os.path.join(DERIVED, "candidates.csv"))
    cal_all = cand[cand["split"] == "cal"]
    anchors = sorted(cal_all["anchor_rid"].unique().tolist())
    random.Random(SEED).shuffle(anchors)
    cal = cal_all[cal_all["anchor_rid"].isin(set(anchors[:3000]))].copy()
    need = set(cal["anchor_rid"]) | set(cal["cand_rid"])
    recs = {r["rid"]: r for r in jl(os.path.join(DERIVED, "records.jsonl")) if r["rid"] in need}
    cal["b2"] = [b2(recs[a], recs[c]) for a, c in zip(cal["anchor_rid"], cal["cand_rid"])]
    b0, b1 = th["logistic"]["b0"], th["logistic"]["b1"]
    cal["p"] = 1 / (1 + np.exp(-(b0 + b1 * cal["b2"])))
    t_l, t_u = th["thresholds"]["T_L"], th["thresholds"]["T_U"]
    cal["region"] = np.where(cal["p"] >= t_u, "accept", np.where(cal["p"] < t_l, "reject", "gray"))
    assert int((cal["region"] == "gray").sum()) == th["calibration_regions"]["gray"]
    return cal.reset_index(drop=True)


def name_blocks_and_dups(df):
    """Name block key and duplicate-key flag for every pair (records.jsonl)."""
    need = set(df["anchor_rid"]) | set(df["cand_rid"])
    recs = {}
    keycount = Counter()
    for r in jl(os.path.join(DERIVED, "records.jsonl")):
        keycount[(r["person"], r["wave"])] += 1
        if r["rid"] in need:
            recs[r["rid"]] = r
    a = [recs[i] for i in df["anchor_rid"]]
    c = [recs[i] for i in df["cand_rid"]]
    block = [f'{x["first"]}|{x["patronym"]}|{x["sex"]}|{w}' for x, w in zip(a, df["wave_a"])]
    dup_anchor = {rid for rid, x in zip(df["anchor_rid"], a) if keycount[(x["person"], x["wave"])] > 1}
    dup_anchor |= {rid for rid, y in zip(df["anchor_rid"], c) if keycount[(y["person"], y["wave"])] > 1}
    return block, df["anchor_rid"].isin(dup_anchor).to_numpy()


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    th = json.load(open(os.path.join(RESULTS, "thresholds.json")))
    t_l, t_u = th["thresholds"]["T_L"], th["thresholds"]["T_U"]
    assert not th["criterion_5_no_gray_zone"]

    cand_lab = pd.read_csv(os.path.join(DERIVED, "candidates.csv"),
                           usecols=["pair_id", "split", "label"])
    cal_lab = dict(zip(cand_lab.loc[cand_lab["split"] == "cal", "pair_id"],
                       cand_lab.loc[cand_lab["split"] == "cal", "label"]))
    del cand_lab                                  # test labels never kept from here
    gray_cal_ids = [r["pair_id"] for r in jl(os.path.join(DERIVED, "gray_cal.jsonl"))]
    lab_gray_cal = {i: int(cal_lab[i]) for i in gray_cal_ids}

    fs = pd.read_json(os.path.join(RESULTS, "fs_decisions.jsonl"), lines=True)
    fs_dec = {s: dict(zip(g["pair_id"], g["D_fs"])) for s, g in fs.groupby("split")}
    fs_p = {s: dict(zip(g["pair_id"], g["p_fs"])) for s, g in fs.groupby("split")}

    if args.dry_run:
        test = load_cal_sample(th)
        y = test["label"].to_numpy()                  # calibration labels only
        rng = random.Random(1)
        llm_cal = {i: rng.randint(0, 1) for i in gray_cal_ids}
        llm_test = llm_cal
        fs_test = fs_dec["cal"]
        fs_p_test = fs_p["cal"]
        llm_info = {"dry_run": "random decisions stand in for the LLM"}
    else:
        raw_path = os.path.join(RESULTS, "raw_qwen_gray.jsonl")
        if not os.path.exists(raw_path):
            sys.exit("results/raw_qwen_gray.jsonl missing: run Task 6 first (score_gray.py)")
        raw = jl(raw_path)
        llm_cal, info_cal = llm_decisions(raw, "cal")
        llm_test, info_test = llm_decisions(raw, "test")
        test = pd.read_csv(os.path.join(DERIVED, "test_sample.csv"))
        gt_ids = {r["pair_id"] for r in jl(os.path.join(DERIVED, "gray_test.jsonl"))}
        assert set(llm_cal) <= set(gray_cal_ids) and set(llm_test) <= gt_ids
        assert info_cal["pairs"] == len(gray_cal_ids) and info_test["pairs"] == len(gt_ids), \
            "LLM output does not cover every gray pair"
        labs = {r["pair_id"]: r["label"] for r in jl(os.path.join(DERIVED, "labels_test.jsonl"))}
        y = test["pair_id"].map(labs).to_numpy()
        assert not np.isnan(y.astype(float)).any()
        fs_test, fs_p_test = fs_dec["test"], fs_p["test"]
        llm_info = {"cal": info_cal, "test": info_test,
                    "meta": json.load(open(os.path.join(RESULTS, "raw_qwen_gray.meta.json")))}

    acc_llm = accuracy(llm_cal, lab_gray_cal)
    acc_fs = accuracy(fs_dec["cal"], lab_gray_cal)

    reg0 = test["region"].to_numpy()
    reg_llm = posterior_regions(test, llm_test, acc_llm, t_l, t_u)
    reg_fs = posterior_regions(test, fs_test, acc_fs, t_l, t_u)
    m_llm, m_fs = metrics(reg0, reg_llm, y), metrics(reg0, reg_fs, y)
    m_b2 = metrics(reg0, reg0, y)

    ci_person, n_persons = cluster_boot(test, reg0, {"LLM": reg_llm, "FS": reg_fs}, y, "anchor_person")

    # criteria
    R = m_llm["R"]
    crit = {
        "1_R_below_0.20": R is None or R < R_MIN,
        "2_CI_lower_le_0": ci_person["LLM"] is None or ci_person["LLM"][0] <= 0,
        "3_false_match_gt_alpha": m_llm["false_match_accepted"] is not None and m_llm["false_match_accepted"] > ALPHA,
        "4_missed_match_gt_beta": m_llm["missed_match_rejected"] is not None and m_llm["missed_match_rejected"] > BETA,
        "5_no_gray_zone": bool(th["criterion_5_no_gray_zone"]),
    }
    h2 = ci_person["LLM_minus_FS"] is not None and ci_person["LLM_minus_FS"][0] > 0
    verdict = "KILLED" if any(crit.values()) else ("POSITIVE" if h2 else "MIXED")

    # sensitivity
    block, dupflag = name_blocks_and_dups(test)
    t2 = test.assign(name_block=block)
    ci_block, n_blocks = cluster_boot(t2, reg0, {"LLM": reg_llm, "FS": reg_fs}, y, "name_block")
    keep = ~dupflag
    m_nodup = metrics(reg0[keep], reg_llm[keep], y[keep])
    m_nodup_fs = metrics(reg0[keep], reg_fs[keep], y[keep])
    aw = test.assign(g0=(reg0 == "gray"), g1=(reg_llm == "gray")).groupby("anchor_rid")[["g0", "g1"]].mean()
    anchor_weighted = {"L_B2": float(aw["g0"].mean()), "L_hyb": float(aw["g1"].mean()),
                       "R": float(1 - aw["g1"].mean() / aw["g0"].mean()) if aw["g0"].mean() > 0 else None}

    # descriptive
    summ2 = json.load(open(os.path.join(DERIVED, "candidates_summary.json")))["summary"]
    blk = summ2["blocking_recall_all"]
    pos = y == 1
    e2e = {"blocking_lost_true_links_all_waves": blk["denominator"] - blk["found"],
           "blocking_recall_all_waves": blk["recall"],
           "sample_positives": int(pos.sum()),
           "sample_positives_auto_rejected_hybrid": int((pos & (reg_llm == "reject")).sum()),
           "sample_positives_auto_accepted_hybrid": int((pos & (reg_llm == "accept")).sum()),
           "sample_positives_left_gray_hybrid": int((pos & (reg_llm == "gray")).sum())}
    e2e["recall_of_auto_accept_hybrid"] = round(e2e["sample_positives_auto_accepted_hybrid"] / max(1, e2e["sample_positives"]), 4)
    per_wave = {}
    for w, g in test.groupby("wave_a"):
        ix = g.index.to_numpy()
        mm = metrics(reg0[ix], reg_llm[ix], y[ix])
        per_wave[str(w)] = {"pairs": int(len(ix)), "L_B2": round(mm["L_B2"], 4), "L_hyb": round(mm["L_hyb"], 4),
                            "false_match_accepted": mm["false_match_accepted"],
                            "missed_match_rejected": mm["missed_match_rejected"]}

    # exploratory H2: D_FS threshold by maximal Youden J on the calibration gray zone
    cal_ids = list(lab_gray_cal)
    cand_t = sorted({fs_p["cal"][i] for i in cal_ids})
    best = max(cand_t, key=lambda t: (
        sum(1 for i in cal_ids if fs_p["cal"][i] >= t and lab_gray_cal[i] == 1) / max(1, sum(lab_gray_cal.values()))
        + sum(1 for i in cal_ids if fs_p["cal"][i] < t and lab_gray_cal[i] == 0) / max(1, len(cal_ids) - sum(lab_gray_cal.values())) - 1))
    fsx_cal = {i: int(fs_p["cal"][i] >= best) for i in cal_ids}
    fsx_test = {i: int(v >= best) for i, v in fs_p_test.items()}
    acc_fsx = accuracy(fsx_cal, lab_gray_cal)
    reg_fsx = posterior_regions(test, fsx_test, acc_fsx, t_l, t_u)
    m_fsx = metrics(reg0, reg_fsx, y)
    ci_fsx, _ = cluster_boot(test, reg0, {"LLM": reg_llm, "FS": reg_fsx}, y, "anchor_person")

    res = {
        "task": "gray-zone-triage Task 7", "dry_run": args.dry_run,
        "verdict": verdict, "criteria_triggered": crit, "H2_registered_holds": h2,
        "primary": {"LLM": m_llm, "FS": m_fs, "B2_alone": m_b2,
                    "ci95_anchor_persons": ci_person, "n_anchor_persons": n_persons},
        "second_stage_accuracy_cal_gray": {"LLM": acc_llm, "FS": acc_fs},
        "llm_output": llm_info,
        "sensitivity": {"ci95_name_blocks": ci_block, "n_name_blocks": n_blocks,
                        "without_duplicate_key_anchors": {"LLM": m_nodup, "FS": m_nodup_fs,
                                                          "pairs_dropped": int(dupflag.sum())},
                        "anchor_weighted": anchor_weighted},
        "descriptive": {"end_to_end": e2e, "per_wave_pair": per_wave},
        "exploratory_H2_fs_youden": {"threshold_p_fs": best, "accuracy_cal": acc_fsx, "FS": m_fsx,
                                     "ci95_anchor_persons": ci_fsx},
        "git_order": None if args.dry_run else git_order_ok(),
    }
    if args.dry_run:
        print(json.dumps({k: res[k] for k in ("verdict", "criteria_triggered", "primary")},
                         indent=1, default=float))
        print("dry run: nothing written")
        return
    json.dump(res, open(os.path.join(RESULTS, "primary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("verdict", "criteria_triggered", "H2_registered_holds", "primary")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
