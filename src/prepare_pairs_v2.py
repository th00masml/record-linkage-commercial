"""v2 pair sets: larger, randomly sampled, leak-controlled, with a disjoint dev split.

Written after the preregistered v1 run (prepare_pairs.py), to fix design problems
found in v1 (see RESULTS.md, "Problems found in the v1 pair set"):

  * v1 took the first 100 person ids / name keys in lexicographic order (all 100
    negatives have first names starting with "A"). v2 samples at random (seed 0).
  * v1 positives were adjacent appearances of a person, negatives any two waves:
    census-year gap alone separates the classes (AUC 0.83). v2 draws negatives from
    the SAME wave pair as a positive, so census years carry no class signal.
  * v1 negatives could have record A in the later wave (positives never). v2
    always puts the earlier wave in A.
  * v1 showed `partner` = the partner's expert-curated person id, i.e. a feature
    derived from the gold linkage. v2 shows the partner's NAME, looked up through
    `source_partner` (a row id within the same census wave), which is evidence a
    genealogist would actually have.
  * v1 matched negatives on lowercased first name + patronym, so 20 negatives
    differed in the full name (middle names) while positives almost never did.
    v2 blocks both classes on the identical normalized full name + sex and
    birth year within 1, so the name carries no signal for any pair.

Candidate space ("block"): every pair of labelled records (r1 in wave w1, r2 in a
later wave w2) with the same normalized full name, same sex, |birth-year diff| <= 1.
A pair is positive iff both rows carry the same expert person id. This is the
blocked candidate set a real linkage system would have to classify.

Splits are by name key (sha1 hash), so no name, and hence no person, appears in
both dev and test. Fellegi-Sunter / logistic weights are fitted on dev only.

Outputs (data/derived/v2/):
  dev_adjacent.jsonl        balanced, consecutive census waves, wave-matched
  test_adjacent.jsonl       same design, disjoint names
  test_longgap.jsonl        balanced, wave pairs >= 20 years apart, wave-matched
  test_natural.jsonl        uniform sample of the adjacent-wave block: the class
                            mix a linker actually faces (not balanced)
  llm_adjacent.jsonl        stratified subset of test_adjacent for LLM scoring
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import random

csv.field_size_limit(10**7)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data/raw")
OUTDIR = os.path.join(ROOT, "data/derived/v2")
SEX = {"Karl": "male", "Kona": "female", "KArl": "male"}
# 1703 and 1729 are far from everything else and thinly labelled (1729).
WAVES = ["1801", "1816", "1835", "1840", "1845", "1850", "1855", "1860",
         "1870", "1880", "1890", "1901", "1910", "1920"]
ADJACENT = list(zip(WAVES, WAVES[1:]))
LONGGAP = [(a, b) for i, a in enumerate(WAVES) for b in WAVES[i + 1:]
           if int(b) - int(a) >= 20]


def norm_name(s):
    return " ".join((s or "").lower().split())


def to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def load_lookup(fn, key, *cols):
    with open(os.path.join(RAW, fn), encoding="utf-8") as f:
        return {r[key]: {c: r.get(c, "") for c in cols} for r in csv.DictReader(f)}


def load_records():
    parishes = load_lookup("parishes.csv", "id", "full_name", "lat", "lon")
    counties = load_lookup("counties.csv", "id", "name")
    with open(os.path.join(RAW, "people.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    name_by_row = {r["id"]: r["nafn_norm"].strip() for r in rows}
    wave_by_row = {r["id"]: r["heimild"].strip() for r in rows}
    recs = []
    for r in rows:
        person = r["person"].strip()
        wave = r["heimild"].strip()
        name = norm_name(r["nafn_norm"])
        by = to_int(r["birthyear"])
        if not person or person.lower() == "nan" or wave not in WAVES or not name or by is None:
            continue
        sp = r["source_partner"].strip()
        partner_name = name_by_row.get(sp, "") if sp and wave_by_row.get(sp) == wave else ""
        par = parishes.get(r["parish"].strip(), {})
        recs.append({
            "row_id": r["id"], "person": person, "wave": wave,
            "name": r["nafn_norm"].strip(), "name_key": name,
            "sex": SEX.get(r["sex"].strip(), r["sex"].strip()),
            "birthyear": by, "status": r["status"].strip(),
            "marriage": r["marriagestatus"].strip(),
            "parish": par.get("full_name", "") or r["parish"].strip(),
            "county": counties.get(r["county"].strip(), {}).get("name", ""),
            "lat": par.get("lat", ""), "lon": par.get("lon", ""),
            "farm_id": r["farm"].strip(), "partner_name": partner_name,
        })
    return recs


def split_of(name_key, sex, dev_frac=0.5):
    h = int(hashlib.sha1(f"{name_key}|{sex}".encode()).hexdigest()[:8], 16)
    return "dev" if (h % 1000) / 1000.0 < dev_frac else "test"


def candidate_space(recs, wave_pairs):
    """{(w1, w2): {"pos": [...], "neg": [...]}} of (r1, r2), r1 in the earlier wave."""
    idx = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in recs:
        idx[x["wave"]][(x["name_key"], x["sex"])].append(x)
    out = {}
    for w1, w2 in wave_pairs:
        A, B = idx[w1], idx[w2]
        pos, neg = [], []
        for k in sorted(A.keys() & B.keys()):
            for r1 in A[k]:
                for r2 in B[k]:
                    if abs(r1["birthyear"] - r2["birthyear"]) <= 1:
                        (pos if r1["person"] == r2["person"] else neg).append((r1, r2))
        out[(w1, w2)] = {"pos": pos, "neg": neg}
    return out


def by_split(space, split):
    keep = lambda pr: split_of(pr[0]["name_key"], pr[0]["sex"]) == split
    return {wp: {c: [p for p in v[c] if keep(p)] for c in ("pos", "neg")}
            for wp, v in space.items()}


def balanced_wave_matched(space, n_pos, rng):
    """n_pos positives sampled uniformly from the pooled positives, then for each one
    a negative from the same wave pair (without replacement). One positive per person."""
    pool = [(wp, pr) for wp, v in sorted(space.items()) for pr in v["pos"]]
    rng.shuffle(pool)
    negs = {wp: rng.sample(v["neg"], len(v["neg"])) for wp, v in sorted(space.items())}
    seen, out = set(), []
    for wp, pr in pool:
        if len(out) >= 2 * n_pos:
            break
        if pr[0]["person"] in seen or not negs[wp]:
            continue
        seen.add(pr[0]["person"])
        out.append((pr, 1))
        out.append((negs[wp].pop(), 0))
    return out


def natural(space, n, rng):
    pool = [(pr, lab) for wp, v in sorted(space.items())
            for lab, c in ((1, "pos"), (0, "neg")) for pr in v[c]]
    return rng.sample(pool, min(n, len(pool)))


def write(name, labelled, rng):
    os.makedirs(OUTDIR, exist_ok=True)
    rng.shuffle(labelled)
    path = os.path.join(OUTDIR, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for pid, ((a, b), lab) in enumerate(labelled):
            f.write(json.dumps({"pid": pid, "a": a, "b": b, "label": lab},
                               ensure_ascii=False) + "\n")
    n_pos = sum(l for _, l in labelled)
    print(f"{name:16} {len(labelled):6} pairs  positives {n_pos:5}  "
          f"prevalence {n_pos / max(1, len(labelled)):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=2000, help="positives per balanced set")
    ap.add_argument("--n-natural", type=int, default=5000)
    ap.add_argument("--n-llm-pos", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    recs = load_records()
    print(f"labelled records in waves {WAVES[0]}-{WAVES[-1]}: {len(recs)}")
    adj = candidate_space(recs, ADJACENT)
    lg = candidate_space(recs, LONGGAP)
    for nm, sp in (("adjacent", adj), ("long-gap", lg)):
        p = sum(len(v["pos"]) for v in sp.values())
        n = sum(len(v["neg"]) for v in sp.values())
        print(f"{nm} block: {p} positive, {n} negative pairs, prevalence {p / (p + n):.3f}")

    dev_adj, test_adj, test_lg = by_split(adj, "dev"), by_split(adj, "test"), by_split(lg, "test")
    write("dev_adjacent", balanced_wave_matched(dev_adj, args.n_pos, rng), rng)
    test_set = balanced_wave_matched(test_adj, args.n_pos, rng)
    write("test_adjacent", list(test_set), rng)
    write("test_longgap", balanced_wave_matched(test_lg, args.n_pos, rng), rng)
    write("test_natural", natural(test_adj, args.n_natural, rng), rng)
    # LLM subset: keep pos/neg partners together so it stays wave-matched.
    pairs_of_two = [test_set[i:i + 2] for i in range(0, len(test_set), 2)]
    llm = [x for two in rng.sample(pairs_of_two, min(args.n_llm_pos, len(pairs_of_two)))
           for x in two]
    write("llm_adjacent", llm, rng)


if __name__ == "__main__":
    main()
