"""Evidence for within-wave positive pairs OUTSIDE the parent-conflict stratum.
Kin links are mostly missing here, so the evidence is: patronym stems, partner
conflict with spouse status, distance on census day, visitor/absence markers,
and whether the ID carries two rows in other waves too (parallel lineage)."""
import itertools, json, os, re, sys, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence import load, km, lab

SPOUSE = re.compile(r"\b(kona hans|hans kona|húsfreyja|húsmóðir|bóndi|húsbóndi|húsráðandi|maður hennar|hennar maður|eiginkona|eiginmaður|ekkja|ekkill|búandi)\b", re.I)
VISIT = re.compile(r"stadd|heimil|gestur|fjarver|ferð|vertíð|sjóróðr|til lækn", re.I)

def stem(s):
    s = unicodedata.normalize("NFKD", (s or "").lower().strip())
    s = "".join(c for c in s if not unicodedata.combining(c))
    for suf in ("sdottir", "dottir", "sson", "son"):
        if s.endswith(suf): return s[: -len(suf)]
    return s

def stems_differ(a, b):
    """True only when the father-name stems share fewer than 3 leading letters
    (Eyjulfs/Eyjolfs, Sigurds/Sigurdar count as the same)."""
    x, y = stem(a), stem(b)
    if not x or not y: return None
    n = 0
    for c1, c2 in zip(x, y):
        if c1 != c2: break
        n += 1
    return n < 3

def all_pairs(df):
    L = df[df.person.map(lab)]
    d = L[L.duplicated(["person", "heimild"], keep=False)]
    out = []
    for (p, w), g in d.groupby(["person", "heimild"], sort=True):
        for a, b in itertools.combinations(g.itertuples(index=False), 2):
            out.append((p, w, a.id, b.id))
    return out, d

def evidence2(df, par, dups, p, w, ida, idb):
    ix = evidence2.__dict__.setdefault("ix", df.set_index("id", drop=False))
    A, B = ix.loc[ida], ix.loc[idb]
    waves_dup = sorted(set(dups[dups.person == p].heimild) - {w})
    def side(R):
        return {"row": R.id, "name": R.nafn_norm, "patronym": R.patronym,
                "by": None if R.by != R.by else int(R.by), "status": R.status,
                "marriage": R.marriagestatus, "partner": R.partner, "father": R.father, "mother": R.mother,
                "parish": par.loc[R.parish].full_name if R.parish in par.index else R.parish, "farm": R.farm,
                "spouse_status": bool(SPOUSE.search(R.status)), "visit_mark": bool(VISIT.search(R.status))}
    a, b = side(A), side(B)
    return {"person": p, "wave": w, "a": a, "b": b,
            "km": km(par, A.parish, B.parish), "same_farm": A.farm == B.farm,
            "by_diff": None if a["by"] is None or b["by"] is None else abs(a["by"] - b["by"]),
            "stems_differ": stems_differ(A.patronym, B.patronym),
            "partner_conflict": bool(A.partner and B.partner and A.partner != B.partner),
            "parallel_waves": waves_dup}
