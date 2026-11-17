"""Task 2 (gray-zone-triage plan): build adjacent-wave homonym candidate pairs and
the calibration/test split from the raw ICE-ID release.

Pre-registered plan: input/ideas/gray-zone-triage-plan.md (Data Sources, Task 2).
Run from anywhere:  python output/data-analysis/make_candidates.py

ASSUMPTIONS (stated before running; checked in code where checkable)
A1  Record fields are built exactly as experiment 1 does (src/prepare_pairs.py):
    same csv.DictReader string parsing, same `by()` birth-year parser
    (int(float(v)), anything else -> None), same SEX map (Karl->male,
    Kona->female, other values passed through), first name / patronym
    stripped + lower-cased, parish -> parishes.full_name (fallback raw parish
    id), lat/lon from parishes.csv, county -> counties.name.
    CHECK: every experiment-1 record (pairs.jsonl sides a/b) is rebuilt from
    the raw file and compared field by field.
A2  Eligibility = non-empty `person` (not "nan"), non-empty normalised first
    name, patronym and sex, integer birth year (by() not None). Experiment 1
    did not require sex; the plan does, so it is an extra filter here.
A3  Waves = distinct `heimild` values, sorted numerically; wave pairs are
    consecutive in that list (w_i, w_{i+1}). Expected 16 waves (checked,
    reported, not enforced).
A4  Exclusion: every eligible record whose (person, wave) occurs as side a or b
    of any experiment-1 pair is removed from the pool (as anchor AND as
    candidate). "Eligible" in all later steps means after exclusion.
A5  Blocking: same normalised first, patronym, sex and |birthyear diff| <= 2.
    Label = (anchor.person == cand.person). No rebalancing, no capping.
A6  Split: distinct anchor person ids (strings, as in the experiment-1 schema),
    sorted lexicographically as strings, shuffled with random.Random(0);
    first floor(n/2) -> "cal", rest -> "test". Seed 0 (frozen).
A7  `rid` = people.csv `id`, assumed a unique integer per row (checked).
A8  Blocking recall denominator = eligible records in w_i (i < last) whose
    person has >= 1 eligible record in w_{i+1}; numerator = those for which at
    least one such record is among their candidates. Reported both over all
    eligible w_i records (primary) and over kept anchors only.
Labels are used here only to count positives/negatives and for blocking recall.
No B2 score is summarised (only checked to run), so nothing is calibrated here.
"""
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

T0 = time.time()
csv.field_size_limit(10**7)
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))          # hybrid-census-linkage
REPO = os.path.dirname(PROJ)                           # record-linkage-commercial
RAW = os.path.join(PROJ, "input", "data", "raw")
DERIVED = os.path.join(PROJ, "input", "data", "derived")
EXP1_PAIRS = os.path.join(REPO, "data", "derived", "pairs.jsonl")
SEED = 0
MAX_DY = 2
BIG = 50 * 1024 * 1024

sys.path.insert(0, os.path.join(REPO, "src"))
from prepare_pairs import SEX, by  # noqa: E402  frozen exp-1 helpers
from baseline import b2  # noqa: E402  frozen rule, unchanged

FIELDS = ["person", "wave", "name", "first", "patronym", "sex", "birthyear",
          "status", "marriage", "parish", "county", "lat", "lon", "farm_id",
          "partner_id"]


def load_lookup(fn, key, *cols):  # verbatim logic of prepare_pairs.load_lookup
    d = {}
    for r in csv.DictReader(open(os.path.join(RAW, fn), encoding="utf-8")):
        d[r[key]] = {c: r.get(c, "") for c in cols}
    return d


def build(r, parishes, counties):
    """Record dict exactly as in src/prepare_pairs.py::main."""
    fn, pat, yr = r["first_name"].strip(), r["patronym"].strip(), by(r["birthyear"])
    par = parishes.get(r["parish"].strip(), {})
    return {
        "person": r["person"].strip(), "wave": r["heimild"].strip(),
        "name": r["nafn_norm"].strip() or f"{fn} {pat}",
        "first": fn.lower(), "patronym": pat.lower(),
        "sex": SEX.get(r["sex"].strip(), r["sex"].strip()),
        "birthyear": yr, "status": r["status"].strip(),
        "marriage": r["marriagestatus"].strip(),
        "parish": par.get("full_name", "") or r["parish"].strip(),
        "county": counties.get(r["county"].strip(), {}).get("name", ""),
        "lat": par.get("lat", ""), "lon": par.get("lon", ""),
        "farm_id": r["farm"].strip(), "partner_id": r["partner"].strip(),
    }


def main():
    os.makedirs(DERIVED, exist_ok=True)
    rep = {"deviations_and_data_notes": []}
    note = rep["deviations_and_data_notes"]

    parishes = load_lookup("parishes.csv", "id", "full_name", "lat", "lon")
    counties = load_lookup("counties.csv", "id", "name")
    par_nocoord = sorted(k for k, v in parishes.items()
                         if not v["lat"].strip() or not v["lon"].strip())

    # ---- load + filter -------------------------------------------------------
    drops = Counter()
    n_rows = 0
    ids_seen = set()
    sex_raw = Counter()
    exp1_idx = defaultdict(list)       # (person, wave) -> recs passing exp-1 filter
    elig = []                          # eligible recs (pre-exclusion), with rid
    rec_parish_unmatched = 0
    rec_county_unmatched = 0
    rec_nocoord = 0
    for r in csv.DictReader(open(os.path.join(RAW, "people.csv"), encoding="utf-8")):
        n_rows += 1
        rid = int(r["id"])
        assert rid not in ids_seen, f"duplicate id {rid}"
        ids_seen.add(rid)
        p = r["person"].strip()
        if not p or p.lower() == "nan":
            drops["1_person_empty"] += 1
            continue
        rec = build(r, parishes, counties)
        sex_raw[r["sex"].strip()] += 1
        if rec["first"] and rec["patronym"] and rec["birthyear"] is not None:
            exp1_idx[(rec["person"], rec["wave"])].append(rec)
        if not rec["first"]:
            drops["2_first_empty"] += 1
            continue
        if not rec["patronym"]:
            drops["3_patronym_empty"] += 1
            continue
        if not rec["sex"]:
            drops["4_sex_empty"] += 1
            continue
        if rec["birthyear"] is None:
            drops["5_birthyear_not_int"] += 1
            continue
        if r["parish"].strip() and r["parish"].strip() not in parishes:
            rec_parish_unmatched += 1
        if r["county"].strip() and r["county"].strip() not in counties:
            rec_county_unmatched += 1
        if rec["lat"] == "" or rec["lon"] == "":
            rec_nocoord += 1
        rec["rid"] = rid
        elig.append(rec)
    n_labelled = n_rows - drops["1_person_empty"]
    rep["load"] = {
        "people_rows": n_rows, "labelled_rows": n_labelled,
        "label_rate": round(n_labelled / n_rows, 4),
        "sequential_drops": dict(sorted(drops.items())),
        "eligible_before_exclusion": len(elig),
        "sex_values_among_labelled_raw": dict(sex_raw.most_common()),
        "parishes_in_lookup": len(parishes),
        "parishes_without_coordinates": par_nocoord,
        "eligible_recs_parish_id_not_in_lookup": rec_parish_unmatched,
        "eligible_recs_county_id_not_in_lookup": rec_county_unmatched,
        "eligible_recs_without_lat_lon": rec_nocoord,
    }

    # ---- A1 check: rebuild experiment-1 records -------------------------------
    exp1 = [json.loads(l) for l in open(EXP1_PAIRS, encoding="utf-8")]
    excl = set()
    n_side, n_exact, mism = 0, 0, []
    for pr in exp1:
        for side in ("a", "b"):
            x = pr[side]
            excl.add((x["person"], x["wave"]))
            n_side += 1
            cands = exp1_idx.get((x["person"], x["wave"]), [])
            # compare on schema fields only: `rid` is added to the same dicts later
            if any({k: c[k] for k in FIELDS} == x for c in cands):
                n_exact += 1
            else:
                diff = None
                if cands:
                    diff = {k: (x.get(k), cands[0].get(k)) for k in FIELDS
                            if x.get(k) != cands[0].get(k)}
                mism.append({"pid": pr["pid"], "side": side, "found": len(cands),
                             "diff": diff})
    rep["exp1_schema_check"] = {"sides": n_side, "exact_dict_match": n_exact,
                                "mismatches": mism[:20], "n_mismatch": len(mism)}
    assert all(set(pr["a"]) == set(FIELDS) for pr in exp1), "exp-1 schema differs"
    if mism:
        note.append(f"{len(mism)} of {n_side} experiment-1 records could not be "
                    "rebuilt identically from the raw file (see exp1_schema_check).")

    # ---- waves ----------------------------------------------------------------
    waves = sorted({x["wave"] for x in elig}, key=lambda w: int(w))
    all_waves = sorted({k[1] for k in exp1_idx} | {x["wave"] for x in elig}, key=int)
    wpairs = list(zip(waves[:-1], waves[1:]))
    rep["waves"] = waves
    rep["n_waves"] = len(waves)
    if len(waves) != 16:
        note.append(f"{len(waves)} waves among eligible records, plan expects 16.")
    if all_waves != waves:
        note.append(f"waves present only among non-eligible labelled rows: "
                    f"{sorted(set(all_waves) - set(waves))}")

    # ---- exclusion ------------------------------------------------------------
    n_before = len(elig)
    elig = [x for x in elig if (x["person"], x["wave"]) not in excl]
    rep["exclusion"] = {"exp1_person_wave_keys": len(excl),
                        "eligible_records_dropped": n_before - len(elig),
                        "eligible_after_exclusion": len(elig)}

    df = pd.DataFrame(elig)
    df["w"] = df["wave"].astype(int)
    per_wave = df.groupby("w").size()
    rep["eligible_per_wave"] = {str(k): int(v) for k, v in per_wave.items()}

    # duplicates of one person within a wave
    dup = df.groupby(["person", "w"]).size()
    dupc = dup[dup > 1]
    rep["duplicates_person_within_wave"] = {
        "person_wave_keys_with_gt1_record": int(len(dupc)),
        "records_involved": int(dupc.sum()),
        "max_records_one_key": int(dupc.max()) if len(dupc) else 1,
        "per_wave": {str(k): int(v) for k, v in
                     dupc.groupby(level="w").size().items()},
    }
    if len(dupc):
        note.append(f"{len(dupc)} (person, wave) keys have >1 eligible record "
                    f"({int(dupc.sum())} records); kept as-is, an anchor can then "
                    "have >1 positive candidate and one person >1 anchor per wave.")

    # ---- candidates (exact blocking via birth-year-shifted merge) -------------
    key = ["first", "patronym", "sex"]
    wnext = dict(zip(waves[:-1], waves[1:]))
    A = df[df["wave"].isin(wnext)][["rid", "person", "w", "wave", "birthyear"] + key].copy()
    A["w_next"] = A["wave"].map(wnext).astype(int)
    C = df[["rid", "person", "w", "birthyear"] + key].rename(
        columns={"rid": "cand_rid", "person": "cand_person", "w": "w_next",
                 "birthyear": "cand_by"})
    parts = []
    for d in range(-MAX_DY, MAX_DY + 1):
        Cd = C.copy()
        Cd["by_join"] = Cd["cand_by"] - d          # anchor.by = cand.by - d
        parts.append(A.merge(Cd, left_on=key + ["w_next", "birthyear"],
                             right_on=key + ["w_next", "by_join"], how="inner"))
    P = pd.concat(parts, ignore_index=True)
    P = P.rename(columns={"rid": "anchor_rid", "person": "anchor_person",
                          "w": "wave_a", "w_next": "wave_b"})
    P["label"] = (P["anchor_person"] == P["cand_person"]).astype(int)
    P = P.sort_values(["wave_a", "anchor_rid", "cand_rid"], kind="mergesort")
    P = P.reset_index(drop=True)
    P["pair_id"] = np.arange(len(P), dtype=np.int64)

    # ---- split ----------------------------------------------------------------
    persons = sorted(set(P["anchor_person"]))
    rnd = random.Random(SEED)
    rnd.shuffle(persons)
    half = len(persons) // 2
    split = {p: ("cal" if i < half else "test") for i, p in enumerate(persons)}
    P["split"] = P["anchor_person"].map(split)

    # ---- blocking recall --------------------------------------------------------
    pw = set(zip(df["person"], df["w"]))
    A["has_true_next"] = [(p, wn) in pw for p, wn in zip(A["person"], A["w_next"])]
    hit = set(P.loc[P["label"] == 1, "anchor_rid"])
    A["hit"] = A["rid"].isin(hit)
    kept = set(P["anchor_rid"])
    A["kept"] = A["rid"].isin(kept)

    def recall(sub):
        den = int(sub["has_true_next"].sum())
        num = int((sub["has_true_next"] & sub["hit"]).sum())
        return {"denominator": den, "found": num,
                "recall": round(num / den, 4) if den else None}

    # ---- summary ---------------------------------------------------------------
    def counts(sub):
        npc = sub.groupby("anchor_rid").size()
        return {"anchors": int(sub["anchor_rid"].nunique()),
                "anchor_persons": int(sub["anchor_person"].nunique()),
                "pairs": int(len(sub)), "positives": int(sub["label"].sum()),
                "negatives": int((sub["label"] == 0).sum()),
                "cand_per_anchor_mean": round(float(npc.mean()), 3) if len(npc) else None,
                "cand_per_anchor_median": float(npc.median()) if len(npc) else None,
                "cand_per_anchor_max": int(npc.max()) if len(npc) else None}

    summ = {"overall": counts(P), "per_split": {}, "per_wave_pair": {},
            "per_split_wave_pair": {}}
    for s in ("cal", "test"):
        summ["per_split"][s] = counts(P[P["split"] == s])
    for wa, wb in wpairs:
        k = f"{wa}-{wb}"
        sub = P[P["wave_a"] == int(wa)]
        c = counts(sub)
        c["eligible_anchor_wave_records"] = int((A["wave"] == wa).sum())
        c["blocking_recall_all"] = recall(A[A["wave"] == wa])
        c["blocking_recall_kept_anchors"] = recall(A[(A["wave"] == wa) & A["kept"]])
        summ["per_wave_pair"][k] = c
        for s in ("cal", "test"):
            summ["per_split_wave_pair"][f"{s}:{k}"] = counts(sub[sub["split"] == s])
    summ["blocking_recall_all"] = recall(A)
    summ["blocking_recall_kept_anchors"] = recall(A[A["kept"]])
    summ["anchors_without_candidates"] = int((~A["kept"]).sum())
    summ["anchors_with_gt1_positive"] = int(
        (P[P["label"] == 1].groupby("anchor_rid").size() > 1).sum())
    summ["split_persons"] = {"cal": half, "test": len(persons) - half}
    rep["summary"] = summ

    # ---- records ---------------------------------------------------------------
    used = set(P["anchor_rid"]) | set(P["cand_rid"])
    recs_out = [x for x in elig if x["rid"] in used]
    recs_out.sort(key=lambda x: x["rid"])

    # ---- unit checks (fail loudly) ---------------------------------------------
    assert not any((x["person"], x["wave"]) in excl for x in recs_out), "excluded key leaked"
    assert P.groupby("anchor_person")["split"].nunique().max() == 1, "person in both splits"
    assert set(P["split"]) <= {"cal", "test"} and P["split"].notna().all()
    byrid = {x["rid"]: x for x in recs_out}
    ar = [byrid[i] for i in P["anchor_rid"]]
    cr = [byrid[i] for i in P["cand_rid"]]
    for a, c, wa, wb in zip(ar, cr, P["wave_a"], P["wave_b"]):
        assert a["first"] == c["first"] and a["patronym"] == c["patronym"] \
            and a["sex"] == c["sex"] and abs(a["birthyear"] - c["birthyear"]) <= MAX_DY \
            and int(a["wave"]) == wa and int(c["wave"]) == wb and wnext[a["wave"]] == c["wave"], \
            "blocking rule violated"
    assert (P["label"].values == np.array([a["person"] == c["person"] for a, c in zip(ar, cr)])).all()
    assert P["pair_id"].is_unique and P["pair_id"].min() == 0 \
        and P["pair_id"].max() == len(P) - 1, "pair_id not contiguous"
    assert not P.duplicated(["anchor_rid", "cand_rid"]).any(), "duplicate pair"
    # completeness: brute-force recount for a seeded sample of anchors
    rs = np.random.RandomState(SEED)
    grp = defaultdict(list)
    for x in elig:
        grp[(x["wave"], x["first"], x["patronym"], x["sex"])].append(x)
    npa = P.groupby("anchor_rid").size()
    elig_by_rid = {x["rid"]: x for x in elig}
    for i in rs.choice(len(A), size=min(2000, len(A)), replace=False):
        row = A.iloc[i]
        a = elig_by_rid[int(row["rid"])]
        bf = sum(abs(c["birthyear"] - a["birthyear"]) <= MAX_DY
                 for c in grp[(wnext[a["wave"]], a["first"], a["patronym"], a["sex"])])
        assert bf == int(npa.get(a["rid"], 0)), f"candidate set incomplete for {a['rid']}"
    sample = rs.choice(len(P), size=min(1000, len(P)), replace=False)
    for i in sample:
        v = b2(ar[i], cr[i])
        assert isinstance(v, float) and math.isfinite(v), "b2 failed"
    rep["unit_checks"] = "all passed (exclusion, split disjointness, blocking rule, "\
        "label, pair_id contiguity, no duplicate pairs, brute-force completeness on "\
        "2000 anchors, b2 runs on 1000 sampled pairs)"

    # ---- write -----------------------------------------------------------------
    with open(os.path.join(DERIVED, "records.jsonl"), "w", encoding="utf-8") as f:
        for x in recs_out:
            f.write(json.dumps({k: x[k] for k in FIELDS + ["rid"]}, ensure_ascii=False) + "\n")
    P[["pair_id", "anchor_rid", "cand_rid", "anchor_person", "wave_a", "wave_b",
       "split", "label"]].to_csv(os.path.join(DERIVED, "candidates.csv"), index=False)
    with open(os.path.join(DERIVED, "split.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, sort_keys=True)

    sizes = {fn: os.path.getsize(os.path.join(DERIVED, fn))
             for fn in ("records.jsonl", "candidates.csv", "split.json")}
    rep["file_sizes_bytes"] = sizes
    big = [fn for fn, s in sizes.items() if s > BIG]
    rep["gitignored_large_files"] = big
    if big:
        gi = os.path.join(PROJ, ".gitignore")
        cur = open(gi, encoding="utf-8").read()
        add = [f"input/data/derived/{fn}" for fn in big if f"input/data/derived/{fn}" not in cur]
        if add:
            with open(gi, "a", encoding="utf-8") as f:
                f.write("\n# Task 2 derived files > 50 MB; deterministic, regenerate with "
                        "output/data-analysis/make_candidates.py\n" + "\n".join(add) + "\n")
        note.append(f"files > 50 MB added to .gitignore (regenerable): {big}")
    rep["records_written"] = len(recs_out)
    rep["seed"] = SEED
    rep["wall_time_s"] = round(time.time() - T0, 1)
    with open(os.path.join(DERIVED, "candidates_summary.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)

    # ---- print -----------------------------------------------------------------
    print(json.dumps({k: rep[k] for k in ("load", "exclusion", "n_waves", "waves",
                                          "eligible_per_wave",
                                          "duplicates_person_within_wave")},
                     ensure_ascii=False, indent=1))
    print("exp1 schema check:", rep["exp1_schema_check"]["exact_dict_match"], "/",
          rep["exp1_schema_check"]["sides"], "sides rebuilt identically")
    hdr = f"{'wave pair':11}{'anchors':>9}{'pairs':>10}{'pos':>9}{'neg':>10}{'recall':>8}{'mean':>7}{'max':>6}"
    print(hdr)
    for k, c in summ["per_wave_pair"].items():
        print(f"{k:11}{c['anchors']:>9}{c['pairs']:>10}{c['positives']:>9}{c['negatives']:>10}"
              f"{str(c['blocking_recall_all']['recall']):>8}{str(c['cand_per_anchor_mean']):>7}"
              f"{str(c['cand_per_anchor_max']):>6}")
    for s in ("cal", "test"):
        c = summ["per_split"][s]
        print(f"{s:11}{c['anchors']:>9}{c['pairs']:>10}{c['positives']:>9}{c['negatives']:>10}"
              f"{'':>8}{c['cand_per_anchor_mean']:>7}{c['cand_per_anchor_max']:>6}")
    c = summ["overall"]
    print(f"{'ALL':11}{c['anchors']:>9}{c['pairs']:>10}{c['positives']:>9}{c['negatives']:>10}"
          f"{summ['blocking_recall_all']['recall']:>8}{c['cand_per_anchor_mean']:>7}{c['cand_per_anchor_max']:>6}")
    print("median cand/anchor:", c["cand_per_anchor_median"],
          "| recall (kept anchors):", summ["blocking_recall_kept_anchors"],
          "| recall (all):", summ["blocking_recall_all"])
    print("split persons:", summ["split_persons"], "| anchors w/o candidates:",
          summ["anchors_without_candidates"], "| anchors with >1 positive:",
          summ["anchors_with_gt1_positive"])
    print("files:", sizes, "| gitignored:", big)
    print("notes:", *note, sep="\n  ")
    print(rep["unit_checks"])
    print(f"wall time {rep['wall_time_s']} s")


if __name__ == "__main__":
    main()
