"""Comparison vectors, Fellegi-Sunter (EM and supervised) and a logistic-regression
linker, numpy only. Used by baselines_v2.py and analysis_v1.py.

Comparison vector per pair: one discrete level per field, -1 = missing (a missing
field contributes no evidence, as in Splink's null level).
"""
import math

import numpy as np

FIELDS = ["birthyear", "parish", "county", "farm", "partner", "distance", "marriage"]
N_LEVELS = {"birthyear": 3, "parish": 2, "county": 2, "farm": 2, "partner": 2,
            "distance": 4, "marriage": 2}
MARRIED = {"G", "E", "S", "L"}  # married, widowed, divorced/separated: never back to "Ó"


def haversine_km(a, b):
    try:
        la1, lo1, la2, lo2 = map(math.radians, [float(a["lat"]), float(a["lon"]),
                                               float(b["lat"]), float(b["lon"])])
    except (KeyError, TypeError, ValueError):
        return None
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(min(1.0, math.sqrt(d)))


def _eq(x, y):
    if not x or not y:
        return -1
    return 1 if x == y else 0


def compare(a, b):
    """a must be the earlier record. Returns a list of levels in FIELDS order."""
    dy = abs(a["birthyear"] - b["birthyear"])
    by = 2 if dy == 0 else (1 if dy == 1 else 0)
    partner = _eq(a.get("partner_name", ""), b.get("partner_name", ""))
    km = haversine_km(a, b)
    dist = -1 if km is None else (3 if km <= 1 else 2 if km <= 10 else 1 if km <= 50 else 0)
    ma, mb = a.get("marriage", ""), b.get("marriage", "")
    if not ma or not mb:
        marr = -1
    else:
        marr = 0 if (ma in MARRIED and mb == "Ó") else 1  # 0 = implausible life course
    return [by, _eq(a["parish"], b["parish"]), _eq(a["county"], b["county"]),
            _eq(a["farm_id"], b["farm_id"]), partner, dist, marr]


def comparison_matrix(pairs):
    return np.array([compare(*ordered(p)) for p in pairs], dtype=np.int64)


def ordered(p):
    a, b = p["a"], p["b"]
    return (a, b) if int(a["wave"]) <= int(b["wave"]) else (b, a)


# ---------------------------------------------------------------- Fellegi-Sunter
class FellegiSunter:
    """Conditional-independence model: weight = sum_k log2(m_k(level)/u_k(level))."""

    def __init__(self, m=None, u=None, lam=None):
        self.m, self.u, self.lam = m, u, lam

    def weights(self, G):
        w = np.zeros(len(G))
        for k, f in enumerate(FIELDS):
            g = G[:, k]
            ok = g >= 0
            w[ok] += np.log2(self.m[f][g[ok]] / self.u[f][g[ok]])
        return w

    @classmethod
    def fit_supervised(cls, G, y, alpha=1.0):
        m, u = {}, {}
        for k, f in enumerate(FIELDS):
            L = N_LEVELS[f]
            g = G[:, k]
            m[f] = _freq(g[(y == 1) & (g >= 0)], L, alpha)
            u[f] = _freq(g[(y == 0) & (g >= 0)], L, alpha)
        return cls(m, u, float(y.mean()))

    @classmethod
    def fit_em(cls, G, n_iter=200, tol=1e-8, seed=0):
        """Unsupervised EM (Winkler 1988 style), no labels used."""
        rng = np.random.default_rng(seed)
        m, u = {}, {}
        for f in FIELDS:
            L = N_LEVELS[f]
            # start: matches favour the highest (agreeing) level
            m[f] = np.linspace(1.0, 3.0, L); m[f] /= m[f].sum()
            u[f] = np.linspace(3.0, 1.0, L); u[f] /= u[f].sum()
            u[f] = 0.9 * u[f] + 0.1 * rng.dirichlet(np.ones(L))
        lam, prev = 0.3, -np.inf
        for _ in range(n_iter):
            lm = np.full(len(G), math.log(lam))
            lu = np.full(len(G), math.log(1 - lam))
            for k, f in enumerate(FIELDS):
                g = G[:, k]
                ok = g >= 0
                lm[ok] += np.log(m[f][g[ok]])
                lu[ok] += np.log(u[f][g[ok]])
            mx = np.maximum(lm, lu)
            ll = float((mx + np.log(np.exp(lm - mx) + np.exp(lu - mx))).sum())
            r = 1.0 / (1.0 + np.exp(lu - lm))
            lam = float(np.clip(r.mean(), 1e-6, 1 - 1e-6))
            for k, f in enumerate(FIELDS):
                L = N_LEVELS[f]
                g = G[:, k]
                ok = g >= 0
                m[f] = _wfreq(g[ok], r[ok], L)
                u[f] = _wfreq(g[ok], 1 - r[ok], L)
            if ll - prev < tol:
                break
            prev = ll
        return cls(m, u, lam)

    def to_dict(self):
        return {"lambda": self.lam,
                "m": {f: self.m[f].round(5).tolist() for f in FIELDS},
                "u": {f: self.u[f].round(5).tolist() for f in FIELDS}}


def _freq(g, L, alpha):
    c = np.bincount(g, minlength=L).astype(float) + alpha
    return c / c.sum()


def _wfreq(g, w, L, eps=1e-6):
    c = np.bincount(g, weights=w, minlength=L) + eps
    return c / c.sum()


# ---------------------------------------------------------------- logistic regression
def one_hot(G):
    cols = []
    for k, f in enumerate(FIELDS):
        for lev in range(-1, N_LEVELS[f]):
            cols.append((G[:, k] == lev).astype(float))
    return np.column_stack(cols)


def fit_logistic(X, y, l2=1.0, n_iter=50):
    """L2-regularised logistic regression by Newton-Raphson; intercept unpenalised."""
    X1 = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X1.shape[1])
    P = np.eye(X1.shape[1]) * l2
    P[0, 0] = 0.0
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-X1 @ w))
        g = X1.T @ (p - y) + P @ w
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + P
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict_logistic(w, X):
    return np.column_stack([np.ones(len(X)), X]) @ w  # log-odds; ranking only


# ---------------------------------------------------------------- metrics
def average_precision(labels, scores):
    """Area under the precision-recall curve (step-wise, ties broken pessimistically)."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    order = np.lexsort((y, -s))  # descending score; among ties, negatives first
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    n_pos = y.sum()
    return float((prec * y).sum() / n_pos) if n_pos else float("nan")


def bootstrap_ci(metric, labels, scores, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(labels), np.asarray(scores)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() in (0, len(i)):
            continue
        out.append(metric(y[i], s[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
