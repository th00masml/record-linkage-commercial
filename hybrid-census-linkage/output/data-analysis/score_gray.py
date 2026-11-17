"""Task 6 (gray-zone-triage plan): score gray-zone pairs with qwen2.5:14b via Ollama.

Run on a machine with Ollama and the model pulled:

    ollama pull qwen2.5:14b
    python hybrid-census-linkage/output/data-analysis/score_gray.py cal
    python hybrid-census-linkage/output/data-analysis/score_gray.py test

Frozen by the pre-registered plan (do not change):
- prompt: experiment 1's SYSTEM + "\\n\\n" + prompt(a, b) from src/score.py
- k = 5 samples per pair, seeds 1..5, temperature 0.7, top_p 1.0, num_predict 12
- score per sample = src/common.py::parse_score (None if unparsed)

Input:  input/data/derived/gray_{cal,test}.jsonl  (pair_id, a, b; no labels)
Output: output/data-analysis/results/raw_qwen_gray.jsonl, one line per (pair, seed):
        {"split", "pair_id", "seed", "raw", "score"}
        output/data-analysis/results/raw_qwen_gray.meta.json (Ollama version, model digest)
Resume-safe: (split, pair_id, seed) already in the output file are skipped.
"""
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(PROJECT)
sys.path.insert(0, os.path.join(REPO, "src"))

from common import parse_score  # noqa: E402  (frozen parser, experiment 1)
from score import SYSTEM, prompt  # noqa: E402  (frozen prompt, experiment 1)

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = "qwen2.5:14b"
SEEDS = [1, 2, 3, 4, 5]
OPTIONS = {"temperature": 0.7, "top_p": 1.0, "num_predict": 12}

DERIVED = os.path.join(PROJECT, "input", "data", "derived")
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "raw_qwen_gray.jsonl")
META = os.path.join(RESULTS, "raw_qwen_gray.meta.json")


def _get(path):
    with urllib.request.urlopen(OLLAMA + path, timeout=30) as r:
        return json.load(r)


def _post(path, body, timeout=180):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def generate(text, seed):
    body = {"model": MODEL, "prompt": text, "stream": False,
            "options": dict(OPTIONS, seed=seed)}
    return _post("/api/generate", body)["response"]


def write_meta():
    version = _get("/api/version").get("version")
    digest = None
    for m in _get("/api/tags").get("models", []):
        if m.get("name") == MODEL or m.get("model") == MODEL:
            digest = m.get("digest")
    if digest is None:
        sys.exit(f"{MODEL} not found in `ollama list`; run `ollama pull {MODEL}` first")
    meta = {}
    if os.path.exists(META):
        meta = json.load(open(META))
        if meta.get("model_digest") not in (None, digest):
            sys.exit(f"model digest changed ({meta['model_digest']} -> {digest}); "
                     "log a deviation before continuing")
    meta.update({"ollama_version": version, "model": MODEL, "model_digest": digest,
                 "seeds": SEEDS, "options": OPTIONS})
    json.dump(meta, open(META, "w"), indent=2)
    print(f"Ollama {version}, {MODEL} digest {digest[:12]}")


def main(split):
    if split not in ("cal", "test"):
        sys.exit("usage: score_gray.py cal|test")
    os.makedirs(RESULTS, exist_ok=True)
    write_meta()
    pairs = [json.loads(l) for l in open(os.path.join(DERIVED, f"gray_{split}.jsonl"))]
    assert all("label" not in p for p in pairs), "gray files must not carry labels"
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            done.add((r["split"], r["pair_id"], r["seed"]))
    todo = [(p, s) for p in pairs for s in SEEDS if (split, p["pair_id"], s) not in done]
    print(f"{split}: {len(pairs)} pairs x {len(SEEDS)} seeds, {len(done)} cached, {len(todo)} to do")
    t0 = time.time()
    with open(OUT, "a") as f:
        for i, (p, seed) in enumerate(todo):
            out = generate(SYSTEM + "\n\n" + prompt(p["a"], p["b"]), seed)
            f.write(json.dumps({"split": split, "pair_id": p["pair_id"], "seed": seed,
                                "raw": out[:24], "score": parse_score(out)}) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                rate = (time.time() - t0) / (i + 1)
                print(f"  [{i+1}/{len(todo)}] {rate:.2f}s/call, "
                      f"~{rate * (len(todo) - i - 1) / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
