"""Protocol 3: score four methods on the within-wave task under ICE-ID labels,
audited labels (primary) and rule-cleaned labels (secondary)."""
import json, math, os, random, sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evidence import load, evidence, lab
from evidence2 import all_pairs, evidence2, stem, stems_differ, SPOUSE, VISIT
import run_audit, run_audit2

SEED = 0
EVER = {"G", "E", "S"}
STATUS_TYPES = [("child", r"barn|son|dóttir|sonur|börn|sön|datter"), ("servant", r"vinnu|hjú|léttadr|léttast|tökub|smali"),
                ("head", r"bóndi|húsbóndi|húsráð|huusbond|búandi"), ("spouse", r"kona hans|hans kona|húsfreyja|húsmóðir")]
import re
STATUS_RX = [(n, re.compile(p, re.I)) for n, p in STATUS_TYPES]


def stype(s):
    for n, rx in STATUS_RX:
        if rx.search(s or ""):
            return n
    return "other"


def hav(a, b):
    if a is None or b is None:
        return None
    la1, lo1, la2, lo2 = map(math.radians, [*a, *b])
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(min(1, math.sqrt(h)))


def build():
    df, par = load()
    ll = {}
    for pid, r in par.iterrows():
        try:
            ll[pid] = (float(r.lat), float(r.lon))
        except ValueError:
            pass
    L = df[df.person.map(lab)].copy()
    L["fn"] = L.first_name.str.strip().str.lower()
    L["pn"] = L.patronym.map(stem)
    rows = {r.id: r for r in L.itertuples(index=False)}

    positives, dups = all_pairs(df)
    # negatives: two per positive, same wave and block, different person
    blk2, blk1 = defaultdict(list), defaultdict(list)
    for r in L.itertuples(index=False):
        if r.fn:
            blk2[(r.heimild, r.fn, r.pn)].append(r.id)
            blk1[(r.heimild, r.fn)].append(r.id)
    rng = random.Random(SEED)
    pairs, ymeta = [], []
    no_neg = 0
    for (p, w, ida, idb) in positives:
        pairs.append((ida, idb)); ymeta.append(("pos", p, w))
        a = rows[ida]
        cand = [x for x in blk2.get((w, a.fn, a.pn), []) if rows[x].person != p]
        if len(cand) < 2:
            cand = [x for x in blk1.get((w, a.fn), []) if rows[x].person != p]
        if not cand:
            no_neg += 1
            continue
        for x in rng.sample(cand, min(2, len(cand))):
            pairs.append((ida, x)); ymeta.append(("neg", p, w))

    def feats(ia, ib):
        a, b = rows[ia], rows[ib]
        dby = abs(a.by - b.by) if a.by == a.by and b.by == b.by else 5.0
        km = hav(ll.get(a.parish), ll.get(b.parish))
        def rel(x, y):
            return (1.0 if x == y else -1.0) if x and y else 0.0
        ma, mb = a.marriagestatus.strip(), b.marriagestatus.strip()
        sd = stems_differ(a.patronym, b.patronym)
        return [min(dby, 5.0), float(a.nafn_norm.strip() == b.nafn_norm.strip()),
                float(a.parish == b.parish), float(a.county == b.county), float(a.farm == b.farm),
                math.log1p(km) if km is not None else 0.0, float(km is None),
                rel(a.partner, b.partner), rel(a.father, b.father), rel(a.mother, b.mother),
                float(ma == mb and ma != ""), float((ma == "Ó" and mb in EVER) or (mb == "Ó" and ma in EVER)),
                float(bool(sd)), float(stype(a.status) == stype(b.status)),
                float(bool(VISIT.search(a.status) or VISIT.search(b.status)))], km, dby

    X, KM, DBY = [], [], []
    for ia, ib in pairs:
        f, km, dby = feats(ia, ib)
        X.append(f); KM.append(km); DBY.append(dby)
    X = np.array(X)
    y_ice = np.array([1 if m[0] == "pos" else 0 for m in ymeta])
    groups = np.array([rows[ia].person for ia, _ in pairs])

    # methods
    S = {}
    S["name_age"] = 2 * X[:, 1] + (X[:, 0] == 0) + 0.5 * (X[:, 0] == 1)
    b2 = []
    for (ia, ib), km, dby in zip(pairs, KM, DBY):
        a, b = rows[ia], rows[ib]
        s = 2.0 if dby == 0 else (1.0 if dby <= 1 else 0.0)
        s += 1.5 * (a.parish == b.parish) + 0.5 * (a.county == b.county) + 1.0 * (a.farm == b.farm)
        s += 2.0 * bool(a.partner and a.partner == b.partner)
        if km is not None:
            s += max(0.0, 1.0 - km / 100.0)
        b2.append(s)
    S["B2"] = np.array(b2)
    for name, mk in [("LR_noisy", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                     ("GB_noisy", lambda: HistGradientBoostingClassifier(random_state=SEED))]:
        oof = np.zeros(len(y_ice))
        for tr, te in GroupKFold(n_splits=5).split(X, y_ice, groups):
            m = mk().fit(X[tr], y_ice[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        S[name] = oof

    return dict(df=df, par=par, rows=rows, positives=positives, dups=dups, pairs=pairs, ymeta=ymeta, X=X, KM=KM, DBY=DBY, y_ice=y_ice, groups=groups, S=S, no_neg=no_neg)


def main(extra=None, tag=""):
    """extra: function(build_dict) -> {method: scores} added to the four protocol-3 methods.
    tag: suffix for output files, so protocol-3 outputs are never overwritten."""
    B = build()
    df, par, rows, positives, dups, pairs, ymeta, X, KM, DBY, y_ice, groups, S, no_neg = (B[k] for k in ['df', 'par', 'rows', 'positives', 'dups', 'pairs', 'ymeta', 'X', 'KM', 'DBY', 'y_ice', 'groups', 'S', 'no_neg'])
    if extra:
        S = {**S, **extra(B)}
    key = {frozenset(pr): i for i, pr in enumerate(pairs) if ymeta[i][0] == "pos"}

    # ---- primary: audited sample
    aud = []
    for fn, stratum, wgt in [("audit_sample.jsonl", "a1", 859 / 50), ("audit2_sample.jsonl", "a2", 9429 / 100)]:
        for l in open(os.path.join(HERE, fn)):
            e = json.loads(l)
            i = key[frozenset((e["a"]["row"], e["b"]["row"]))]
            lab_a = {"over_merge": 0, "probable_over_merge": 0, "label_correct": 1, "duplicate_row": 1}.get(e["final"])
            if lab_a is None:
                continue  # undetermined: dropped from every condition
            aud.append((i, 1, lab_a, stratum, wgt))
    neg_idx = [i for i, m in enumerate(ymeta) if m[0] == "neg"]
    for i in random.Random(SEED).sample(neg_idx, 300):
        aud.append((i, 0, 0, "neg", 2 * 10288 / 300))
    idx = np.array([a[0] for a in aud]); yI = np.array([a[1] for a in aud]); yA = np.array([a[2] for a in aud])
    st = np.array([a[3] for a in aud]); w = np.array([a[4] for a in aud])

    def metrics(s, yI, yA, w):
        out = {}
        p, r, t = precision_recall_curve(yI, s, sample_weight=w)
        f1 = 2 * p * r / np.maximum(p + r, 1e-12)
        thr = t[np.argmax(f1[:-1])]
        for nm, y in [("ice", yI), ("clean", yA)]:
            pred = s >= thr
            tp = (w * (pred & (y == 1))).sum(); fp = (w * (pred & (y == 0))).sum(); fn_ = (w * (~pred & (y == 1))).sum()
            prec, rec = tp / max(tp + fp, 1e-12), tp / max(tp + fn_, 1e-12)
            out[nm] = {"auc": roc_auc_score(y, s, sample_weight=w), "ap": average_precision_score(y, s, sample_weight=w),
                       "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12)}
        return out

    def boot(fn_metric, strata, B=1000, seed=0):
        r = np.random.default_rng(seed)
        groups_ = {g: np.where(strata == g)[0] for g in np.unique(strata)}
        res = []
        for _ in range(B):
            take = np.concatenate([r.choice(ix, len(ix)) for ix in groups_.values()])
            try:
                res.append(fn_metric(take))
            except ValueError:
                continue
        return np.array(res)

    primary = {"n_pairs": len(aud), "n_audited_pos_kept": int((st != "neg").sum()),
               "audited_pos_relabelled_negative": int(((st != "neg") & (yA == 0)).sum()), "methods": {}}
    for name, s_all in S.items():
        s = s_all[idx]
        m = metrics(s, yI, yA, w)
        bs = boot(lambda t: (roc_auc_score(yI[t], s[t], sample_weight=w[t]), roc_auc_score(yA[t], s[t], sample_weight=w[t])), st)
        d = bs[:, 1] - bs[:, 0]
        m["auc_ci"] = {"ice": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), "clean": np.percentile(bs[:, 1], [2.5, 97.5]).tolist()}
        m["delta_auc"] = {"est": m["clean"]["auc"] - m["ice"]["auc"], "ci": np.percentile(d, [2.5, 97.5]).tolist()}
        primary["methods"][name] = m
    for cond in ("ice", "clean"):
        primary[f"ranking_{cond}"] = sorted(S, key=lambda n: -primary["methods"][n][cond]["auc"])

    # ---- secondary: rule-cleaned labels on all pairs (partly circular)
    conflict = {tuple(c) for c in json.load(open(os.path.join(HERE, "conflict_pairs.json")))}
    y_rule = y_ice.copy()
    flagged = Counter()
    for (p, w_, ida, idb) in positives:
        i = key[frozenset((ida, idb))]
        if (p, w_, ida, idb) in conflict:
            v = run_audit.verdict(evidence(df, par, p, w_, ida, idb))
        else:
            v = run_audit2.verdict(evidence2(df, par, dups, p, w_, ida, idb))[0]
        flagged[v] += 1
        if v == "over_merge":
            y_rule[i] = 0
    secondary = {"n_pairs": len(y_ice), "positives_flagged": dict(flagged), "methods": {}}
    ones = np.ones(len(y_ice))
    for name, s in S.items():
        m = metrics(s, y_ice, y_rule, ones)
        bs = boot(lambda t: (roc_auc_score(y_ice[t], s[t]), roc_auc_score(y_rule[t], s[t])), np.zeros(len(y_ice)), B=200)
        m["delta_auc"] = {"est": m["clean"]["auc"] - m["ice"]["auc"], "ci": np.percentile(bs[:, 1] - bs[:, 0], [2.5, 97.5]).tolist()}
        secondary["methods"][name] = m
    for cond in ("ice", "clean"):
        secondary[f"ranking_{cond}"] = sorted(S, key=lambda n: -secondary["methods"][n][cond]["auc"])

    np.savez(os.path.join(HERE, f"noise_effect_scores{tag}.npz"), idx=idx, yI=yI, yA=yA, st=st, w=w,
             y_ice=y_ice, y_rule=y_rule, **{f"S_{k}": v for k, v in S.items()})
    out = {"positives": len(positives), "pairs": len(pairs), "positives_without_negative": no_neg,
           "primary_audited": primary, "secondary_rule_cleaned": secondary}
    json.dump(out, open(os.path.join(HERE, f"noise_effect_results{tag}.json"), "w"), indent=1, default=float)

    for title, blk in [("PRIMARY: audited sample", primary), ("SECONDARY: rule-cleaned, all pairs", secondary)]:
        print(f"\n{title}  (n={blk['n_pairs']})")
        print(f"{'method':10} {'AUC ice':>8} {'AUC clean':>9} {'dAUC [95% CI]':>24} {'AP ice':>7} {'AP cln':>7} {'F1 ice':>7} {'F1 cln':>7} {'R ice':>6} {'R cln':>6}")
        for n, m in blk["methods"].items():
            ci = m["delta_auc"]["ci"]
            print(f"{n:12} {m['ice']['auc']:8.3f} {m['clean']['auc']:9.3f} {m['delta_auc']['est']:+8.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] "
                  f"{m['ice']['ap']:7.3f} {m['clean']['ap']:7.3f} {m['ice']['f1']:7.3f} {m['clean']['f1']:7.3f} {m['ice']['recall']:6.3f} {m['clean']['recall']:6.3f}")
        print("ranking ICE-ID:", blk["ranking_ice"], "| ranking clean:", blk["ranking_clean"])
    print({k: v for k, v in out.items() if not isinstance(v, dict)}, primary["n_audited_pos_kept"], primary["audited_pos_relabelled_negative"], secondary["positives_flagged"])


if __name__ == "__main__":
    main()
