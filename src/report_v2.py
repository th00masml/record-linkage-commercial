"""Table of all v2 results (outputs/v2) plus any LLM runs on v2 sets
(outputs/v2_<set>__*/scores_*.json). Writes outputs/v2/summary.json."""
import glob
import json
import os

from common import paired_delta_ci

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "outputs")
SETS = ["test_adjacent", "test_longgap", "test_natural", "llm_adjacent"]
METHODS = ["B1", "B2", "FS_em", "FS_sup", "FS_splink", "LR"]


def main():
    summary = {}
    for s in SETS:
        print(f"\n== {s}")
        rows = {}
        for m in METHODS:
            p = os.path.join(OUT, "v2", f"scores_{s}_{m}.json")
            if os.path.exists(p):
                rows[m] = json.load(open(p))
        for d in sorted(glob.glob(os.path.join(OUT, f"v2_{s}__*", "scores_*.json"))):
            r = json.load(open(d))
            rows[os.path.basename(os.path.dirname(d)).split("__", 1)[1]] = r
        b2 = rows.get("B2")
        summary[s] = {}
        for m, r in rows.items():
            line = f"  {m:28} AUC {r['auc']:.4f} [{r['ci95'][0]:.4f},{r['ci95'][1]:.4f}]"
            if "ap" in r:
                line += f"  AP {r['ap']:.4f}"
            entry = {"auc": r["auc"], "ci95": r["ci95"], "ap": r.get("ap"), "n": len(r["labels"]),
                     "prevalence": sum(r["labels"]) / len(r["labels"])}
            if b2 and m != "B2":
                assert r["pid"] == b2["pid"], f"{m}: pair order differs from B2"
                obs, lo, hi, p = paired_delta_ci(b2["labels"], b2["scores"], r["scores"])
                line += f"  d vs B2 {obs:+.4f} [{lo:+.4f},{hi:+.4f}]"
                entry["delta_vs_B2"] = [obs, lo, hi, p]
            summary[s][m] = entry
            print(line)
    json.dump(summary, open(os.path.join(OUT, "v2", "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
