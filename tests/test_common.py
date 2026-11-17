"""Run from the repo root: python -m pytest tests  (or python tests/test_common.py)."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from common import parse_score, roc_auc  # noqa: E402
from linkage import FellegiSunter, average_precision, compare  # noqa: E402


def brute_auc(y, s):
    pos = [b for b, l in zip(s, y) if l == 1]
    neg = [b for b, l in zip(s, y) if l == 0]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def test_roc_auc_matches_brute_force_with_ties():
    rng = np.random.default_rng(1)
    for _ in range(20):
        y = rng.integers(0, 2, 60)
        s = rng.integers(0, 5, 60).astype(float)  # many ties
        assert abs(roc_auc(y, s) - brute_auc(y, s)) < 1e-12


def test_parse_score():
    cases = {"0.85": 0.85, "1.00": 1.0, "1": 1.0, "0": 0.0, ".5": 0.5, "0.85.": 0.85,
             "Probability: 0.3": 0.3, "**0.72**": 0.72, "10": None, "20.5": None,
             "1.5": None, "": None, "unsure": None}
    for s, want in cases.items():
        assert parse_score(s) == want, (s, parse_score(s))


def test_parse_score_reproduces_cached_v1_scores():
    for fn in ("raw_comm.jsonl", "raw_local.jsonl"):
        for line in open(os.path.join(ROOT, "outputs", fn)):
            r = json.loads(line)
            assert parse_score(r["raw"]) == r["score"], r


def test_average_precision():
    assert average_precision([1, 1, 0, 0], [4, 3, 2, 1]) == 1.0
    assert abs(average_precision([0, 1, 0, 1], [4, 3, 2, 1]) - (0.5 + 0.5) / 2) < 1e-12
    # ties are broken pessimistically (negative ranked first)
    assert average_precision([1, 0], [1, 1]) == 0.5


def _rec(**kw):
    base = {"birthyear": 1800, "wave": "1835", "parish": "P", "county": "C", "farm_id": "1",
            "partner_name": "", "lat": "64.0", "lon": "-21.0", "marriage": "Ó"}
    base.update(kw)
    return base


def test_compare_levels():
    a, b = _rec(), _rec(wave="1840")
    assert compare(a, b) == [2, 1, 1, 1, -1, 3, 1]
    assert compare(_rec(marriage="G"), _rec(wave="1840", marriage="Ó"))[-1] == 0
    assert compare(_rec(lat=""), b)[5] == -1


def test_fellegi_sunter_em_recovers_planted_structure():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.random(n) < 0.4
    G = np.zeros((n, 7), int)
    G[:, 0] = np.where(y, rng.choice(3, n, p=[0.0, 0.2, 0.8]), rng.choice(3, n, p=[0.0, 0.7, 0.3]))
    for k in (1, 2, 3, 4):
        G[:, k] = np.where(y, rng.random(n) < 0.8, rng.random(n) < 0.05)
    G[:, 5] = np.where(y, 3, rng.choice(4, n, p=[0.85, 0.1, 0.03, 0.02]))
    G[:, 6] = 1
    w = FellegiSunter.fit_em(G).weights(G)
    assert roc_auc(y.astype(int), w) > 0.97


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
