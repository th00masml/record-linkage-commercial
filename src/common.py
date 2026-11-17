"""Shared helpers: model calls (Ollama, Claude CLI, Anthropic API), score parsing,
AUC, bootstrap."""
import json
import re
import shutil
import subprocess
import urllib.request

import numpy as np

OLLAMA = "http://localhost:11434"
MODEL = "qwen2.5:14b"


# ---------------------------------------------------------------- local (Ollama)
def generate(prompt, seed=7, num_predict=12, temperature=0.0, timeout=120, model=MODEL):
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed,
                    "num_predict": num_predict, "top_p": 1.0},
    }
    req = urllib.request.Request(
        OLLAMA + "/api/generate", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["response"]


def ollama_metadata(model=MODEL):
    """Ollama server version and the model's digest, for the run log."""
    meta = {"model": model}
    try:
        with urllib.request.urlopen(OLLAMA + "/api/version", timeout=10) as r:
            meta["ollama_version"] = json.load(r).get("version")
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=10) as r:
            for m in json.load(r).get("models", []):
                if m.get("name") == model or m.get("model") == model:
                    meta["digest"] = m.get("digest")
                    meta["details"] = m.get("details")
    except OSError as e:
        meta["error"] = str(e)
    return meta


# ---------------------------------------------------------------- commercial (CLI)
class CLIError(RuntimeError):
    pass


def generate_cli(prompt, system=None, model="claude-sonnet-5", timeout=90,
                 system_mode="append", json_output=False):
    """Call the authenticated Claude CLI in print mode. No temperature control.

    system_mode="append" adds `system` to Claude Code's default system prompt (what
    the v1 run did); "replace" uses `system` as the whole system prompt, which is
    the cleaner instrument for new runs. With json_output=True returns
    (text, model_ids) parsed from --output-format json."""
    cmd = ["claude", "-p", "--model", model, "--allowedTools", ""]
    if system:
        cmd += ["--system-prompt" if system_mode == "replace" else "--append-system-prompt", system]
    if json_output:
        cmd += ["--output-format", "json"]
    cmd += [prompt]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        # Before: a failed call returned "" and was silently imputed as 0.5.
        raise CLIError(f"claude exited {r.returncode}: {(r.stderr or r.stdout)[:300]}")
    out = (r.stdout or "").strip()
    if not json_output:
        return out
    d = json.loads(out)
    return str(d.get("result", "")).strip(), sorted((d.get("modelUsage") or {}).keys())


def cli_version():
    if not shutil.which("claude"):
        return None
    r = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=30)
    return r.stdout.strip()


# ---------------------------------------------------------------- commercial (API)
def generate_api(prompt, system=None, model="claude-sonnet-5", temperature=0.0, max_tokens=16):
    """Anthropic Messages API with explicit temperature. Needs ANTHROPIC_API_KEY and
    `pip install anthropic`. Returns (text, exact model id reported by the API)."""
    import anthropic  # optional dependency
    client = anthropic.Anthropic()
    kw = {"system": system} if system else {}
    msg = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "user", "content": prompt}], **kw)
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return text.strip(), msg.model


# ---------------------------------------------------------------- parsing and metrics
_NUM = re.compile(r"(?<![\d.])(1(?:\.0+)?|0?\.\d+|0)(?!\.?\d)")


def parse_score(s):
    """First standalone number in [0, 1]. Returns None if there is none.

    The v1 regex would read "10" as 1 and "0.5" inside "20.5" as 0.5; the lookarounds
    reject numbers embedded in longer numbers. All 400 cached v1 outputs parse to the
    same value under both (checked in tests/test_common.py)."""
    m = _NUM.search((s or "").strip())
    return float(m.group(1)) if m else None


def roc_auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    u = ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def bootstrap_auc_ci(labels, scores, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    n = len(labels)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        lb = labels[idx]
        if lb.sum() in (0, len(lb)):
            continue
        out.append(roc_auc(lb, scores[idx]))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_delta_ci(labels, scores_a, scores_b, n_boot=2000, seed=0):
    """AUC(b) - AUC(a) on the same items: observed delta, percentile 95% CI, and the
    share of bootstrap deltas <= 0 (a one-sided bootstrap proportion, not a p-value
    from a formal test)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    sa, sb = np.asarray(scores_a), np.asarray(scores_b)
    obs = roc_auc(labels, sb) - roc_auc(labels, sa)
    n = len(labels)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        lb = labels[idx]
        if lb.sum() in (0, len(lb)):
            continue
        d.append(roc_auc(lb, sb[idx]) - roc_auc(lb, sa[idx]))
    d = np.array(d)
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())
