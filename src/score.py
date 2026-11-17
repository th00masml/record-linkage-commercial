"""Score record pairs for same-person with a model. Cached / resume-safe.

v1 (preregistered, reproduces the cached outputs exactly):
    python score.py local          # qwen2.5:14b via Ollama
    python score.py comm           # claude-sonnet-5 via the authenticated CLI

New runs (v2 pairs, few-shot, repeats, temperature-controlled API):
    python score.py comm --pairs v2/llm_adjacent --backend api --repeats 3
    python score.py local --pairs v2/llm_adjacent --shots 4
    python score.py comm --pairs v2/llm_adjacent --backend cli --system-mode replace

Every run writes outputs/<tag>/run_meta.json with the exact model identifiers and
tool versions that answered (Ollama version + model digest, CLI version, or the
model id the API reports), plus the prompt settings.
"""
import argparse
import datetime
import json
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from common import (CLIError, bootstrap_auc_ci, cli_version, generate, generate_api,
                    generate_cli, ollama_metadata, parse_score, roc_auc)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DERIVED = os.path.join(ROOT, "data/derived")
OUT = os.path.join(ROOT, "outputs")

SYSTEM = "You are an expert genealogist performing historical record linkage on Icelandic census data. Reply with only a single number between 0.00 and 1.00, nothing else."
COMM_MODEL = "claude-sonnet-5"


def fmt(r):
    partner = r.get("partner_name") or r.get("partner_id") or "?"
    plabel = "Partner name" if "partner_name" in r else "Partner id"
    lines = [f"  Name: {r['name']}", f"  Sex: {r['sex']}",
             f"  Birth year: {r['birthyear']}", f"  Census year: {r['wave']}",
             f"  Status: {r['status'] or '?'}  Marital: {r['marriage'] or '?'}",
             f"  Parish: {r['parish'] or '?'}  County: {r['county'] or '?'}",
             f"  Farm id: {r['farm_id'] or '?'}  {plabel}: {partner}"]
    return "\n".join(lines)


def records_block(a, b):
    return f"RECORD A:\n{fmt(a)}\n\nRECORD B:\n{fmt(b)}"


def prompt(a, b, examples=()):
    shots = ""
    if examples:
        shots = "Worked examples (with the correct answer):\n\n" + "\n\n".join(
            f"{records_block(e['a'], e['b'])}\nAnswer: {'1.00' if e['label'] else '0.00'}"
            for e in examples) + "\n\nNow the pair to decide.\n\n"
    return f"""Two Icelandic census records. Decide whether they are the SAME individual recorded in two different census years.

Icelandic names are patronymic, so very many different people share the same name and patronym -- the name alone cannot identify a person. Weigh instead: birth-year consistency, parish/region and whether any move between them is geographically plausible, household and partner, and life-course plausibility (age, marital status).

{shots}{records_block(a, b)}

Probability they are the same person (0.00 = certainly different people, 1.00 = certainly the same). Number only."""


def few_shot_examples(k, seed):
    """k/2 positives + k/2 negatives from v2 dev_adjacent (names disjoint from test)."""
    if k <= 0:
        return []
    dev = [json.loads(l) for l in open(os.path.join(DERIVED, "v2/dev_adjacent.jsonl"), encoding="utf-8")]
    rng = random.Random(seed)
    pos = rng.sample([p for p in dev if p["label"] == 1], k // 2 + k % 2)
    neg = rng.sample([p for p in dev if p["label"] == 0], k // 2)
    ex = pos + neg
    rng.shuffle(ex)
    return ex


def run_tag(args):
    if args.pairs == "pairs" and args.shots == 0 and args.repeats == 1 and args.backend in ("ollama", "cli") \
            and args.system_mode == "append":
        return None  # the preregistered v1 configuration: keep v1 file names
    parts = [args.pairs.replace("/", "_"), args.method, args.backend, f"shots{args.shots}"]
    if args.backend == "cli":
        parts.append(f"sys{args.system_mode}")
    return "__".join(parts)


def call(args, pr):
    if args.method == "local":
        return generate(SYSTEM + "\n\n" + pr, num_predict=8 if not args.shots else 12,
                        seed=args.seed), None
    if args.backend == "api":
        return generate_api(pr, system=SYSTEM, model=args.model, temperature=args.temperature)
    if args.json_output:
        text, ids = generate_cli(pr, system=SYSTEM, model=args.model,
                                 system_mode=args.system_mode, json_output=True)
        return text, ",".join(ids)
    return generate_cli(pr, system=SYSTEM, model=args.model, system_mode=args.system_mode), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["local", "comm"])
    ap.add_argument("--pairs", default="pairs", help="file under data/derived without .jsonl")
    ap.add_argument("--backend", choices=["ollama", "cli", "api"], default=None)
    ap.add_argument("--model", default=COMM_MODEL)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1, help="independent samples per pair")
    ap.add_argument("--temperature", type=float, default=0.0, help="api backend only")
    ap.add_argument("--system-mode", choices=["append", "replace"], default="append",
                    help="cli backend: append to or replace Claude Code's system prompt")
    ap.add_argument("--json-output", action="store_true", help="cli: record the model id that answered")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-failures", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1, help="parallel calls")
    args = ap.parse_args()
    args.backend = args.backend or ("ollama" if args.method == "local" else "cli")

    pairs = [json.loads(l) for l in open(os.path.join(DERIVED, f"{args.pairs}.jsonl"), encoding="utf-8")]
    examples = few_shot_examples(args.shots, args.seed)
    tag = run_tag(args)
    outdir = OUT if tag is None else os.path.join(OUT, tag)
    os.makedirs(outdir, exist_ok=True)
    raw_path = os.path.join(outdir, f"raw_{args.method}.jsonl")

    meta = {"started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "args": vars(args), "system_prompt": SYSTEM,
            "few_shot_pids_from_dev": [e["pid"] for e in examples]}
    if args.method == "local":
        meta["ollama"] = ollama_metadata()
    elif args.backend == "cli":
        meta["claude_cli_version"] = cli_version()
    if tag is not None:
        json.dump(meta, open(os.path.join(outdir, "run_meta.json"), "w"), indent=2)

    done = {}
    if os.path.exists(raw_path):
        for l in open(raw_path):
            r = json.loads(l)
            done[(r["pid"], r.get("rep", 0))] = r
    todo = [(p, k) for p in pairs for k in range(args.repeats) if (p["pid"], k) not in done]
    print(f"{args.method}/{args.backend}: {len(pairs)} pairs x {args.repeats}, "
          f"{len(done)} cached, {len(todo)} to do -> {outdir}")
    t0, failures = time.time(), 0
    # Calls are independent, so they may run in parallel (--workers); rows are
    # written by this thread only, in completion order, and keyed by (pid, rep).
    with open(raw_path, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(call, args, prompt(p["a"], p["b"], examples)): (p, k) for p, k in todo}
        for i, fut in enumerate(as_completed(futs)):
            p, k = futs[fut]
            try:
                out, model_id = fut.result()
            except (CLIError, OSError, subprocess.TimeoutExpired) as e:
                failures += 1
                print(f"  pid {p['pid']} rep {k}: {e}")
                if failures >= args.max_failures:
                    for other in futs:
                        other.cancel()
                    raise SystemExit("too many failed calls; fix the instrument and rerun (cached rows are kept)")
                continue
            row = {"pid": p["pid"], "raw": out[:24], "score": parse_score(out)}
            if tag is not None:
                row["rep"] = k
                if model_id:
                    row["model_id"] = model_id
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(todo)}] {time.time()-t0:.0f}s", flush=True)

    rows = [json.loads(l) for l in open(raw_path)]
    by_pid = {}
    for r in rows:
        if r["score"] is not None:
            by_pid.setdefault(r["pid"], []).append(r["score"])
    unparsed = sum(1 for p in pairs if p["pid"] not in by_pid)
    # unparsed / failed pairs are imputed at 0.5 (counted and reported)
    scores = [float(np.mean(by_pid[p["pid"]])) if p["pid"] in by_pid else 0.5 for p in pairs]
    labels = [p["label"] for p in pairs]
    auc = roc_auc(labels, scores)
    lo, hi = bootstrap_auc_ci(labels, scores)
    name = {"local": "M_local_qwen14b", "comm": "M_comm_sonnet5"}[args.method]
    res = {"method": name, "auc": auc, "ci95": [lo, hi], "unparsed": unparsed,
           "pid": [p["pid"] for p in pairs], "labels": labels, "scores": scores}
    if args.repeats > 1:
        per_rep = []
        for k in range(args.repeats):
            sk = {r["pid"]: r["score"] for r in rows if r.get("rep", 0) == k and r["score"] is not None}
            per_rep.append(roc_auc(labels, [sk.get(p["pid"], 0.5) for p in pairs]))
        sd = [float(np.std(by_pid[p["pid"]])) for p in pairs if len(by_pid.get(p["pid"], [])) > 1]
        res["auc_per_repeat"] = per_rep
        res["mean_within_pair_sd"] = float(np.mean(sd)) if sd else None
        res["model_ids_seen"] = sorted({r["model_id"] for r in rows if r.get("model_id")})
    json.dump(res, open(os.path.join(outdir, f"scores_{name}.json"), "w"))
    print(f"{name}: AUC={auc:.4f} [{lo:.4f},{hi:.4f}] unparsed={unparsed}"
          + (f" per-repeat={['%.4f' % a for a in res['auc_per_repeat']]}" if args.repeats > 1 else ""))


if __name__ == "__main__":
    main()
