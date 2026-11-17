"""Merge / split / distinct items from ICE-ID for the look-or-ask agent study.

Every item is two census rows that answer the same name lookup
(first name + patronym + sex, birth year within +-WINDOW). The lookup
window holds exactly those two rows, counting labelled AND unlabelled rows
in the fixture waves, so the ground truth is complete, not just true on the
curated subset.

  merge     two rows, one person, different waves        K = 1
  split     two rows, two persons, different waves       K = 2
            (looks like one person tracked over time; it is two)
  distinct  two rows, two persons, same wave             K = 2
            (the tau-bench "second same-name customer" control)

Persons the curators placed twice in one wave are dropped entirely, and so
is any window that touches one of their rows: those IDs are over-merges
(e.g. person 100123 sits in two parishes with two fathers in 1840-1860).

No model calls. Deterministic (seed 0).
"""
import bisect
import json
import math
import os
import random
from collections import Counter, defaultdict

import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data/raw")
DERIVED = os.path.join(ROOT, "data/derived")

WAVES = [1835, 1840, 1845, 1850, 1855, 1860]  # contiguous, 45-92% labelled
WINDOW = 2          # covers 99.6% of adjacent-wave birth-year drift
N_PER_CLASS = 200
SEED = 0
SEX = {"Karl": "male", "Kona": "female"}


def is_lab(p):
    p = p.strip()
    return bool(p) and p.lower() != "nan"


def km(a, b):
    try:
        la1, lo1, la2, lo2 = map(math.radians, map(float, [a["lat"], a["lon"], b["lat"], b["lon"]]))
    except (TypeError, ValueError):
        return None
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return round(2 * 6371 * math.asin(min(1, math.sqrt(h))), 1)


def same(x, y):
    """True/False when both sides are known, None otherwise."""
    if not x or not y:
        return None
    return x == y


def main():
    os.makedirs(DERIVED, exist_ok=True)
    df = pd.read_csv(os.path.join(RAW, "people.csv"), dtype=str, keep_default_na=False)
    parishes = pd.read_csv(os.path.join(RAW, "parishes.csv"), dtype=str, keep_default_na=False).set_index("id")
    counties = pd.read_csv(os.path.join(RAW, "counties.csv"), dtype=str, keep_default_na=False).set_index("id")

    # partner / father / mother hold PERSON ids (checked: row 2, Ólöf Jónsdóttir,
    # partner 11847 = person 11847 Jón Ásmundsson, bóndi, same farm 1771)
    person_name = {}
    for p, n in zip(df.person, df.nafn_norm):
        if is_lab(p):
            person_name.setdefault(p.strip(), n.strip())

    df["lab"] = df.person.map(is_lab)
    lab = df[df.lab]
    suspect = set(lab[lab.duplicated(["person", "heimild"], keep=False)].person)

    df["w"] = df.heimild.astype(int)
    df["by"] = pd.to_numeric(df.birthyear, errors="coerce")
    db = df[df.w.isin(WAVES) & df.by.notna()
            & df.first_name.str.strip().ne("") & df.patronym.str.strip().ne("")].copy()
    db["key"] = list(zip(db.first_name.str.strip().str.lower(),
                         db.patronym.str.strip().str.lower(),
                         db.sex.map(lambda s: SEX.get(s.strip(), s.strip()))))
    db["bad"] = db.person.isin(suspect)

    def rec(r):
        par = parishes.loc[r.parish] if r.parish in parishes.index else None
        cty = counties.loc[r.county] if r.county in counties.index else None
        return {
            "row": r.id, "wave": int(r.w), "name": r.nafn_norm.strip(),
            "first": r.first_name.strip(), "middle": r.middle_name.strip(),
            "patronym": r.patronym.strip(), "sex": SEX.get(r.sex.strip(), r.sex.strip()),
            "birthyear": int(r.by), "status": r.status.strip(), "marriage": r.marriagestatus.strip(),
            "parish": par["full_name"] if par is not None else r.parish,
            "county": cty["name"] if cty is not None else "",
            "lat": par["lat"] if par is not None else "", "lon": par["lon"] if par is not None else "",
            "farm_id": r.farm.strip(),
            "partner_person": r.partner.strip(), "partner_name": person_name.get(r.partner.strip(), ""),
            "father_person": r.father.strip(), "father_name": person_name.get(r.father.strip(), ""),
            "mother_person": r.mother.strip(), "mother_name": person_name.get(r.mother.strip(), ""),
            "person": r.person.strip(),  # ground truth: never show to the agent
        }

    pools = defaultdict(list)
    drops = Counter()
    for key, g in db.groupby("key", sort=True):
        g = g.sort_values(["by", "id"])
        bys = g.by.tolist()
        rows = list(g.itertuples(index=False))
        n_key = len(rows)
        n_key_unlab = int((~g.lab).sum())
        for i in range(len(rows)):
            hi_i = bisect.bisect_right(bys, bys[i] + WINDOW)
            for j in range(i + 1, hi_i):
                a, b = rows[i], rows[j]
                lo, hi = min(a.by, b.by) - WINDOW, max(a.by, b.by) + WINDOW
                n_win = bisect.bisect_right(bys, hi) - bisect.bisect_left(bys, lo)
                if n_win != 2:
                    drops["window_not_2"] += 1
                    continue
                if not (a.lab and b.lab):
                    drops["unlabelled_in_pair"] += 1
                    continue
                if a.bad or b.bad:
                    drops["suspect_person"] += 1
                    continue
                pa, pb = a.person.strip(), b.person.strip()
                if pa == pb and a.w == b.w:
                    drops["same_person_same_wave"] += 1  # cannot happen after cleaning
                    continue
                cls = "merge" if pa == pb else ("distinct" if a.w == b.w else "split")
                A, B = (rec(a), rec(b)) if (a.w, a.id) <= (b.w, b.id) else (rec(b), rec(a))
                ev = {
                    "by_diff": abs(A["birthyear"] - B["birthyear"]),
                    "wave_gap": B["wave"] - A["wave"],
                    "parish_same": A["parish"] == B["parish"],
                    "county_same": bool(A["county"]) and A["county"] == B["county"],
                    "farm_same": same(A["farm_id"], B["farm_id"]),
                    "km": km(A, B),
                    "father_same": same(A["father_person"], B["father_person"]),
                    "mother_same": same(A["mother_person"], B["mother_person"]),
                    "partner_same": same(A["partner_person"], B["partner_person"]),
                }
                # visible evidence an agent could read off the two records
                if cls == "merge":
                    ev["visible"] = "confirms" if (ev["father_same"] or ev["mother_same"]
                                                   or ev["partner_same"]) else "none"
                else:
                    ev["visible"] = "refutes_same" if (ev["father_same"] is False
                                                       or ev["mother_same"] is False) else "none"
                if cls != "merge" and (ev["farm_same"] or ev["father_same"]
                                       or ev["mother_same"] or ev["partner_same"]):
                    # same name, birth year within 2, and a shared farm or kin under two
                    # IDs: almost surely one human the curators failed to link
                    drops[f"{cls}_probable_undermerge"] += 1
                    continue
                # how much the class label can be trusted
                if cls == "merge":
                    strength = "curator_link"       # a positive expert judgment
                elif cls == "distinct":
                    strength = "same_wave"          # two places in one census: two humans
                elif ev["father_same"] is False or ev["mother_same"] is False:
                    strength = "kin_differs"        # different parents: two humans
                else:
                    strength = "unlinked_only"      # curators did not link; NOT proof of two
                pools[cls].append({
                    "label_strength": strength,
                    "class": cls, "K": 1 if cls == "merge" else 2,
                    "key": {"first": key[0], "patronym": key[1], "sex": key[2]},
                    "lookup_window": [int(lo), int(hi)],
                    "name_universe_n": n_key, "name_universe_unlabelled": n_key_unlab,
                    "a": A, "b": B, "evidence": ev,
                })

    rng = random.Random(SEED)
    split = rng.sample(pools["split"], min(N_PER_CLASS, len(pools["split"])))
    # merges matched to splits cell by cell, so the surface of the two records
    # (time apart, birth-year drift, parish, farm) cannot tell the classes apart
    def cell(x):
        e = x["evidence"]
        return (e["wave_gap"], e["by_diff"], e["parish_same"], bool(e["farm_same"]))
    merge_cells = defaultdict(list)
    for x in pools["merge"]:
        merge_cells[cell(x)].append(x)
    split_cells = defaultdict(list)
    for x in split:
        split_cells[cell(x)].append(x)
    merge, split, unmatched = [], [], 0
    for c in sorted(split_cells):
        s_c, m_c = split_cells[c], merge_cells.get(c, [])
        n = min(len(s_c), len(m_c))
        split += s_c[:n]  # s_c is already a random draw
        merge += rng.sample(m_c, n)
        unmatched += len(s_c) - n  # dropped: no merge with that surface exists
    distinct = rng.sample(pools["distinct"], min(N_PER_CLASS, len(pools["distinct"])))

    items = merge + split + distinct
    rng.shuffle(items)
    for k, x in enumerate(items):
        x["iid"] = k
    with open(os.path.join(DERIVED, "merge_split.jsonl"), "w") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    def desc(xs):
        if not xs:
            return {}
        ev = [x["evidence"] for x in xs]
        return {
            "n": len(xs),
            "wave_gap": dict(sorted(Counter(e["wave_gap"] for e in ev).items())),
            "by_diff": dict(sorted(Counter(e["by_diff"] for e in ev).items())),
            "parish_same": round(sum(e["parish_same"] for e in ev) / len(ev), 3),
            "farm_same": round(sum(bool(e["farm_same"]) for e in ev) / len(ev), 3),
            "visible": dict(Counter(e["visible"] for e in ev)),
            "label_strength": dict(Counter(x["label_strength"] for x in xs)),
            "name_universe_n_median": sorted(x["name_universe_n"] for x in xs)[len(xs) // 2],
        }

    summary = {
        "waves": WAVES, "window": WINDOW, "seed": SEED,
        "suspect_persons_dropped": len(suspect),
        "drops": dict(drops),
        "pool": {c: len(v) for c, v in pools.items()},
        "merge_cells_unmatched": unmatched,
        "pool_desc": {c: desc(v) for c, v in pools.items()},
        "sample_desc": {"merge": desc(merge), "split": desc(split), "distinct": desc(distinct)},
    }
    json.dump(summary, open(os.path.join(DERIVED, "merge_split_summary.json"), "w"),
              indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
