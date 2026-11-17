"""Canonical probabilistic baseline: a Fellegi-Sunter model fitted with Splink
(unsupervised: u by random sampling, m by EM; no labels used).

Trained on every labelled record whose name falls in the DEV split (same hash split
as prepare_pairs_v2.py), then applied to the v2 test sets. Writes
outputs/v2/splink_model.json and outputs/v2/scores_<set>_FS_splink.json.

Requires `pip install -r requirements-splink.txt` (splink, duckdb).
"""
import json
import os
import sys

import splink.comparison_level_library as cll
import splink.comparison_library as cl
from splink import Linker, SettingsCreator, block_on
from splink.backends.duckdb import DuckDBAPI

from common import roc_auc
from linkage import average_precision, bootstrap_ci
from prepare_pairs_v2 import OUTDIR as PAIRDIR
from prepare_pairs_v2 import load_records, split_of

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "outputs/v2")
TEST_SETS = ["test_adjacent", "test_longgap", "test_natural", "llm_adjacent"]
MARRIED = "('G','E','S','L')"


def row(x):
    nz = lambda v: v if v not in ("", None) else None
    flt = lambda v: float(v) if v not in ("", None) else None
    return {"unique_id": int(x["row_id"]), "name_key": x["name_key"], "sex": x["sex"],
            "birthyear": int(x["birthyear"]), "wave": int(x["wave"]),
            "parish": nz(x["parish"]), "county": nz(x["county"]),
            "farm_id": nz(x["farm_id"]), "partner_name": nz(x["partner_name"]),
            "lat": flt(x["lat"]), "lon": flt(x["lon"]), "marriage": nz(x["marriage"])}


def settings():
    birthyear = cl.CustomComparison(
        output_column_name="birthyear",
        comparison_levels=[
            cll.NullLevel("birthyear"),
            cll.ExactMatchLevel("birthyear"),
            cll.CustomLevel("abs(birthyear_l - birthyear_r) <= 1", "within 1 year"),
            cll.ElseLevel(),
        ])
    marriage = cl.CustomComparison(
        output_column_name="marriage",
        comparison_levels=[
            cll.CustomLevel("marriage_l IS NULL OR marriage_r IS NULL", "null").configure(is_null_level=True),
            cll.CustomLevel(
                f"(wave_l < wave_r AND marriage_l IN {MARRIED} AND marriage_r = 'Ó') OR "
                f"(wave_r < wave_l AND marriage_r IN {MARRIED} AND marriage_l = 'Ó')",
                "married then never-married"),
            cll.ElseLevel(),
        ])
    return SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            birthyear,
            cl.ExactMatch("parish"), cl.ExactMatch("county"), cl.ExactMatch("farm_id"),
            cl.ExactMatch("partner_name"),
            cl.DistanceInKMAtThresholds("lat", "lon", [1, 10, 50]),
            marriage,
        ],
        blocking_rules_to_generate_predictions=[
            "l.name_key = r.name_key AND l.sex = r.sex AND l.wave <> r.wave "
            "AND abs(l.birthyear - r.birthyear) <= 1",
        ],
        additional_columns_to_retain=["wave"],
        retain_intermediate_calculation_columns=False,
    )


def train(dev_rows):
    db = DuckDBAPI()
    linker = Linker(db.register(dev_rows, dataset_display_name="dev"), settings())
    linker.training.estimate_u_using_random_sampling(max_pairs=1e7, seed=0)
    # EM over same-name blocks; name_key/sex are not comparisons, so every
    # comparison is estimated in this session.
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("name_key", "sex"), record_sample_proportion=1.0)
    return linker


def main():
    os.makedirs(OUT, exist_ok=True)
    recs = load_records()
    dev_rows = [row(x) for x in recs if split_of(x["name_key"], x["sex"]) == "dev"]
    print(f"training Splink on {len(dev_rows)} dev-split records (no labels)")
    model = train(dev_rows).misc.save_model_to_json(
        os.path.join(OUT, "splink_model.json"), overwrite=True)

    sets = {s: [json.loads(l) for l in open(os.path.join(PAIRDIR, f"{s}.jsonl"), encoding="utf-8")]
            for s in TEST_SETS}
    test_rows = {}
    for pairs in sets.values():
        for p in pairs:
            for x in (p["a"], p["b"]):
                test_rows[int(x["row_id"])] = row(x)
    db = DuckDBAPI()
    linker = Linker(db.register(list(test_rows.values()), dataset_display_name="test"), model)
    pred = linker.inference.predict().as_record_list()
    mw = {(min(r["unique_id_l"], r["unique_id_r"]), max(r["unique_id_l"], r["unique_id_r"])):
          r["match_weight"] for r in pred}
    for name, pairs in sets.items():
        keys = [tuple(sorted((int(p["a"]["row_id"]), int(p["b"]["row_id"])))) for p in pairs]
        missing = sum(k not in mw for k in keys)
        if missing:
            sys.exit(f"{name}: {missing} pairs not produced by the Splink blocking rule")
        scores = [mw[k] for k in keys]
        labels = [p["label"] for p in pairs]
        auc, ap = roc_auc(labels, scores), average_precision(labels, scores)
        res = {"method": "FS_splink", "set": name, "auc": auc,
               "ci95": bootstrap_ci(roc_auc, labels, scores), "ap": ap,
               "ap_ci95": bootstrap_ci(average_precision, labels, scores),
               "pid": [p["pid"] for p in pairs], "labels": labels, "scores": scores}
        json.dump(res, open(os.path.join(OUT, f"scores_{name}_FS_splink.json"), "w"))
        print(f"{name:14} FS_splink AUC={auc:.4f} [{res['ci95'][0]:.4f},{res['ci95'][1]:.4f}] AP={ap:.4f}")


if __name__ == "__main__":
    main()
