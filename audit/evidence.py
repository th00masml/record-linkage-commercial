"""Evidence table for within-wave pairs that share a person ID but have
different father or mother person IDs. Internal evidence only (people.csv):
co-residence of the linked parents, patronym vs father's first name, parent
ages, and which row the person's other-wave appearances continue."""
import itertools, json, math, os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data/raw")

def lab(p): p = p.strip(); return bool(p) and p.lower() != "nan"

def load():
    df = pd.read_csv(os.path.join(RAW, "people.csv"), dtype=str, keep_default_na=False)
    par = pd.read_csv(os.path.join(RAW, "parishes.csv"), dtype=str, keep_default_na=False).set_index("id")
    df["by"] = pd.to_numeric(df.birthyear, errors="coerce")
    return df, par

def km(par, p, q):
    try:
        a, b = par.loc[p], par.loc[q]
        la1, lo1, la2, lo2 = map(math.radians, map(float, [a.lat, a.lon, b.lat, b.lon]))
    except Exception:
        return None
    h = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return round(2*6371*math.asin(min(1, math.sqrt(h))), 1)

def lcp(a, b):
    n = 0
    for x, y in zip(a.lower(), b.lower()):
        if x != y: break
        n += 1
    return n

def patronym_ok(patronym, father_first):
    """Patronym = genitive of the father's first name + son/dóttir.
    Prefix match is a heuristic: Jón/Jónsdóttir, Bjarni/Bjarnason, Sigurður/Sigurðarson."""
    if not patronym or not father_first:
        return None
    stem = patronym.lower()
    for suf in ("sdóttir", "dóttir", "sson", "son"):
        if stem.endswith(suf): stem = stem[: -len(suf)]; break
    f = father_first.lower()
    return lcp(stem, f) >= max(2, min(4, len(f) - 1))

def conflict_pairs(df):
    L = df[df.person.map(lab)]
    d = L[L.duplicated(["person", "heimild"], keep=False)]
    out = []
    for (p, w), g in d.groupby(["person", "heimild"], sort=True):
        for a, b in itertools.combinations(g.itertuples(index=False), 2):
            fa = a.father and b.father and a.father != b.father
            mo = a.mother and b.mother and a.mother != b.mother
            if fa or mo:
                out.append((p, w, a.id, b.id))
    return out

def evidence(df, par, p, w, ida, idb):
    rows = df.set_index("id", drop=False) if not hasattr(evidence, "_ix") else evidence._ix
    evidence._ix = rows
    byp = evidence._byp if hasattr(evidence, "_byp") else df[df.person.map(lab)].groupby("person")
    evidence._byp = byp
    A, B = rows.loc[ida], rows.loc[idb]
    def parent(pid, wave, child):
        if not pid: return None
        g = byp.get_group(pid) if pid in byp.groups else None
        if g is None: return {"person": pid, "found": False}
        same = g[g.heimild == wave]
        r = same.iloc[0] if len(same) else g.iloc[0]
        return {"person": pid, "name": r.nafn_norm, "first": r.first_name, "by": None if pd.isna(r.by) else int(r.by),
                "in_wave": bool(len(same)),
                "coresident": bool(len(same)) and any(same.farm == child.farm),
                "farm": r.farm if len(same) else "", "status": r.status if len(same) else "",
                "age_gap": None if pd.isna(r.by) or pd.isna(child.by) else int(child.by - r.by)}
    def side(R):
        f, m = parent(R.father, w, R), parent(R.mother, w, R)
        return {"row": R.id, "name": R.nafn_norm, "patronym": R.patronym, "by": None if pd.isna(R.by) else int(R.by),
                "status": R.status, "parish": par.loc[R.parish].full_name if R.parish in par.index else R.parish,
                "parish_id": R.parish, "farm": R.farm, "father": f, "mother": m,
                "patronym_ok": patronym_ok(R.patronym, f.get("first") if f else ""),
                "household": df[(df.heimild == w) & (df.farm == R.farm)][["nafn_norm", "birthyear", "status"]]
                             .head(12).values.tolist()}
    # other-wave appearances of this person: which side do they continue?
    other = byp.get_group(p)
    other = other[other.heimild != w].sort_values("heimild")
    cont = []
    for r in other.itertuples(index=False):
        cont.append({"wave": r.heimild, "name": r.nafn_norm, "by": r.birthyear, "status": r.status,
                     "parish": par.loc[r.parish].full_name if r.parish in par.index else r.parish,
                     "father": r.father, "mother": r.mother,
                     "km_to_a": km(par, r.parish, A.parish), "km_to_b": km(par, r.parish, B.parish)})
    return {"person": p, "wave": w, "km_ab": km(par, A.parish, B.parish), "same_farm": A.farm == B.farm,
            "a": side(A), "b": side(B), "other_waves": cont}
