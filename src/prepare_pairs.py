"""Build readable records and the deterministic pair set.
Positive = same person seen in two different census waves (earliest vs latest
= maximum temporal drift). Hard negative = two DIFFERENT people who share the
same first name + patronym + sex and were born within 1 year (Icelandic
homonyms -- Galton's 'little variety of surnames'). No model calls here.
"""
import csv
import json
import os
import random

csv.field_size_limit(10**7)
HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data/raw")
DERIVED = os.path.join(ROOT, "data/derived")
N_PER_CLASS = 100
SEX = {"Karl": "male", "Kona": "female"}


def load_lookup(fn, key, *cols):
    d = {}
    for r in csv.DictReader(open(os.path.join(RAW, fn), encoding="utf-8")):
        d[r[key]] = {c: r.get(c, "") for c in cols}
    return d


def by(v):
    try:
        return int(float(v))
    except Exception:
        return None


def main():
    os.makedirs(DERIVED, exist_ok=True)
    parishes = load_lookup("parishes.csv", "id", "full_name", "lat", "lon")
    counties = load_lookup("counties.csv", "id", "name")

    recs = []
    for r in csv.DictReader(open(os.path.join(RAW, "people.csv"), encoding="utf-8")):
        p = r["person"].strip()
        if not p or p.lower() == "nan":
            continue
        fn, pat, yr = r["first_name"].strip(), r["patronym"].strip(), by(r["birthyear"])
        if not fn or not pat or yr is None:
            continue
        par = parishes.get(r["parish"].strip(), {})
        rec = {
            "person": p, "wave": r["heimild"].strip(),
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
        recs.append(rec)

    # positives
    from collections import defaultdict
    byp = defaultdict(list)
    for x in recs:
        byp[x["person"]].append(x)
    rng = random.Random(0)
    pos = []
    for pid in sorted(byp):
        waves = byp[pid]
        yrs = sorted({int(w["wave"]) for w in waves})
        if len(yrs) < 2:
            continue
        # adjacent census appearances = the smallest wave gap (realistic linkage)
        gap, lo, hi = min((yrs[k + 1] - yrs[k], yrs[k], yrs[k + 1])
                          for k in range(len(yrs) - 1))
        a = next(w for w in waves if int(w["wave"]) == lo)
        b = next(w for w in waves if int(w["wave"]) == hi)
        pos.append((a, b))
        if len(pos) >= N_PER_CLASS:
            break

    # hard negatives: homonyms
    key = defaultdict(list)
    for x in recs:
        key[(x["first"], x["patronym"], x["sex"])].append(x)
    neg = []
    for k in sorted(key):
        v = key[k]
        if len({x["person"] for x in v}) < 2:
            continue
        v2 = sorted(v, key=lambda x: (x["person"], x["wave"]))
        placed = False
        for i in range(len(v2)):
            for j in range(i + 1, len(v2)):
                a, b = v2[i], v2[j]
                if a["person"] != b["person"] and a["wave"] != b["wave"] \
                        and abs(a["birthyear"] - b["birthyear"]) <= 1:
                    neg.append((a, b))
                    placed = True
                    break
            if placed:
                break
        if len(neg) >= N_PER_CLASS:
            break

    pairs = []
    pid = 0
    for a, b in pos:
        pairs.append({"pid": pid, "a": a, "b": b, "label": 1}); pid += 1
    for a, b in neg:
        pairs.append({"pid": pid, "a": a, "b": b, "label": 0}); pid += 1
    rng.shuffle(pairs)
    with open(os.path.join(DERIVED, "pairs.jsonl"), "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"positives {len(pos)}  hard-negatives {len(neg)}  total {len(pairs)}")
    print("sample positive:")
    a, b = pos[0]
    print(" ", a["name"], a["birthyear"], a["wave"], a["parish"], "|", b["wave"], b["parish"])
    print("sample hard-negative:")
    a, b = neg[0]
    print(" ", a["name"], a["birthyear"], a["wave"], a["parish"], "|", b["name"], b["birthyear"], b["wave"], b["parish"])


if __name__ == "__main__":
    main()
