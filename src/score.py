"""Score record pairs for same-person with a model. Usage: python score.py [local|comm]
  local = qwen2.5:14b (Ollama);  comm = claude-sonnet-5 (authenticated CLI).
Cached / resume-safe. Same prompt for both."""
import json
import os
import sys
import time

from common import (bootstrap_auc_ci, generate, generate_cli, parse_score,
                    roc_auc)

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
DERIVED = os.path.join(ROOT, "data/derived")
OUT = os.path.join(ROOT, "outputs")

SYSTEM = "You are an expert genealogist performing historical record linkage on Icelandic census data. Reply with only a single number between 0.00 and 1.00, nothing else."


def fmt(r):
    lines = [f"  Name: {r['name']}", f"  Sex: {r['sex']}",
             f"  Birth year: {r['birthyear']}", f"  Census year: {r['wave']}",
             f"  Status: {r['status'] or '?'}  Marital: {r['marriage'] or '?'}",
             f"  Parish: {r['parish'] or '?'}  County: {r['county'] or '?'}",
             f"  Farm id: {r['farm_id'] or '?'}  Partner id: {r['partner_id'] or '?'}"]
    return "\n".join(lines)


def prompt(a, b):
    return f"""Two Icelandic census records. Decide whether they are the SAME individual recorded in two different census years.

Icelandic names are patronymic, so very many different people share the same name and patronym -- the name alone cannot identify a person. Weigh instead: birth-year consistency, parish/region and whether any move between them is geographically plausible, household and partner, and life-course plausibility (age, marital status).

RECORD A:
{fmt(a)}

RECORD B:
{fmt(b)}

Probability they are the same person (0.00 = certainly different people, 1.00 = certainly the same). Number only."""


def main(method):
    os.makedirs(OUT, exist_ok=True)
    pairs = [json.loads(l) for l in open(os.path.join(DERIVED, "pairs.jsonl"))]
    raw_path = os.path.join(OUT, f"raw_{method}.jsonl")
    done = {}
    if os.path.exists(raw_path):
        for l in open(raw_path):
            r = json.loads(l)
            done[r["pid"]] = r
    f = open(raw_path, "a")
    todo = [p for p in pairs if p["pid"] not in done]
    print(f"{method}: {len(pairs)} pairs, {len(done)} cached, {len(todo)} to do")
    t0 = time.time()
    for i, p in enumerate(todo):
        pr = prompt(p["a"], p["b"])
        if method == "local":
            out = generate(SYSTEM + "\n\n" + pr, num_predict=8)
        else:
            out = generate_cli(pr, system=SYSTEM, model="claude-sonnet-5")
        f.write(json.dumps({"pid": p["pid"], "raw": out[:24],
                            "score": parse_score(out)}) + "\n")
        f.flush()
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(todo)}] {time.time()-t0:.0f}s", flush=True)
    f.close()
    sc = {r["pid"]: r["score"] for r in (json.loads(l) for l in open(raw_path))}
    unparsed = sum(1 for p in pairs if sc.get(p["pid"]) is None)
    scores = [sc[p["pid"]] if sc.get(p["pid"]) is not None else 0.5 for p in pairs]
    labels = [p["label"] for p in pairs]
    auc = roc_auc(labels, scores)
    lo, hi = bootstrap_auc_ci(labels, scores)
    name = {"local": "M_local_qwen14b", "comm": "M_comm_sonnet5"}[method]
    json.dump({"method": name, "auc": auc, "ci95": [lo, hi], "unparsed": unparsed,
               "pid": [p["pid"] for p in pairs], "labels": labels, "scores": scores},
              open(os.path.join(OUT, f"scores_{name}.json"), "w"))
    print(f"{name}: AUC={auc:.4f} [{lo:.4f},{hi:.4f}] unparsed={unparsed}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "local")
