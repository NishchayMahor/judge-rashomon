"""Unattended Arena full run (PREREG-arena.md). Phases, each logged to logs/arena_status:
  rep0   all 6 configs on 3,200 items + pairwise both-orders subset, configs run concurrently (interleaved)
  retry  exactly one retry of transient failures, failed entries moved to cache_arena_failed/
  noise  best pairwise + best pointwise config (selection-half kappa, eligible configs) re-run twice on the measurement half
Prompts, parsing and decoding come from run_judges.py, unchanged. Cache: cache_arena/rep{r}/{cfg}/.
NIM requests are paced to stay under the free-tier rate limit (shared across threads).
Usage: uv run python src/arena_run.py [--phase all|rep0|retry|noise] [--limit N]
"""
import argparse, datetime, hashlib, json, os, re, shutil, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import run_judges as rj
from arena_analyze import Data, eligible, kappa_from, per_pair_conf

ROOT = rj.ROOT
CACHE, FAILED = ROOT / "cache_arena", ROOT / "cache_arena_failed"
STATUS = ROOT / "logs" / "arena_status"
JUDGES = {"deepseek": "deepseek-ai/deepseek-v4.1-flash", "mistral": "mistralai/mistral-large-4-1t-a52b",
          "nemotron": "nvidia/nemotron-3-ultra-550b-a55b"}  # all on NIM (PREREG-arena pre-freeze change 3)
CONFIGS = [f"{j}-{f}" for j in JUDGES for f in ("pair", "point")]
WORKERS = {"deepseek": 8, "mistral": 6, "nemotron": 3}  # nemotron low (deviation 2026-11-11a)  # deepseek is slow (~230 s/call), so more in flight
TRANSIENT = re.compile(r"timed out|Connection error|Error code: (429|5\d\d)", re.I)
RPM = float(os.environ.get("JR_RPM", 35))
EMPTY_RETRIES = 3  # nemotron on NIM: empty reply = transport failure (deviation 2026-11-11a)


def log(msg):
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


class Paced:
    """Wraps the NIM client so request starts are at least 60/RPM s apart across all threads."""
    def __init__(self, cli):
        self._cli, self._lock, self._next = cli, threading.Lock(), 0.0
        self.chat = self
        self.completions = self

    def _one(self, kw):
        with self._lock:
            wait = self._next - time.monotonic()
            self._next = max(self._next, time.monotonic()) + 60 / RPM
        if wait > 0:
            time.sleep(wait)
        return self._cli.chat.completions.create(**kw)

    def create(self, **kw):
        r = self._one(kw)
        if not kw["model"].startswith("nvidia/nemotron"):
            return r
        for k in range(EMPTY_RETRIES):  # same request re-sent; every retry is logged
            if (r.choices[0].message.content or "").strip():
                break
            with (ROOT / "logs" / "nemotron_empty_retries.log").open("a") as f:
                f.write(f"{datetime.datetime.utcnow().isoformat()} empty reply, retry {k + 1}/{EMPTY_RETRIES}, "
                        f"reasoning_tokens={getattr(r.usage.completion_tokens_details, 'reasoning_tokens', None)}\n")
            time.sleep(10 * 2 ** k)
            r = self._one(kw)
        return r


def load_items(limit=None):
    items = json.loads((ROOT / "data" / "arena_items.json").read_text())
    return items[:limit] if limit else items


def tasks(cfg, items, rep, rev=False):
    """(model, provider, sys, user, cache path, nonce). rev = the both-orders subset with the coin flipped."""
    j, fmt = cfg.split("-")
    nonce = "" if rep == 0 else f"\n(run {rep})"
    for it in items:
        if fmt == "pair":
            swap = it["swap"] ^ rev
            a, b = (it["ans_second"], it["ans_first"]) if swap else (it["ans_first"], it["ans_second"])
            p = CACHE / f"rep{rep}" / (cfg + ("-rev" if rev else "")) / f"{it['id']}.json"
            yield JUDGES[j], None, rj.PAIR_SYS, rj.PAIR_USER.format(q=it["prompt"], a=a, b=b), p, nonce
        else:
            for side in ("first", "second"):
                ans = it[f"ans_{side}"]
                h = hashlib.sha1((it["prompt"] + ans).encode()).hexdigest()[:12]
                p = CACHE / f"rep{rep}" / cfg / f"{it['id']}_{h}.json"
                yield JUDGES[j], None, rj.POINT_SYS, rj.POINT_USER.format(q=it["prompt"], a=ans), p, nonce


def verdicts(cfg, items, rep):
    """Canonical int verdicts (see arena_analyze): 0 first, 1 second, 2 tie, -1 not run, -2 failed."""
    out = np.full(len(items), -1)
    ts = list(tasks(cfg, items, rep))
    per = 1 if cfg.endswith("-pair") else 2
    for i, it in enumerate(items):
        fs = [t[4] for t in ts[i * per:(i + 1) * per]]
        if not all(f.exists() for f in fs):
            continue
        texts = [json.loads(f.read_text()).get("text", "") for f in fs]
        if per == 1:
            v = rj.parse_pair(texts[0])
            v = None if v is None else 2 if v == "tie" else (0 if v == "model_a" else 1) ^ it["swap"]
        else:
            s = [rj.parse_point(t) for t in texts]
            v = None if any(x is None or not 1 <= x <= 10 for x in s) else 0 if s[0] > s[1] else 1 if s[1] > s[0] else 2
        out[i] = -2 if v is None else v
    return out


def run_tasks(cfg, ts):
    cli = PACED
    with ThreadPoolExecutor(WORKERS[cfg.split("-")[0]]) as ex:
        res = list(ex.map(lambda t: rj.call(cli, *t), ts))
    return len(res), sum(1 for r in res if r.get("error"))


def run_parallel(jobs):
    """jobs: {name: task list}; run every config at once so calls interleave over time."""
    with ThreadPoolExecutor(len(jobs)) as ex:
        futs = {n: ex.submit(run_tasks, n.replace("-rev", ""), ts) for n, ts in jobs.items()}
        for n, f in futs.items():
            log(f"{n}: {f.result()[0]} calls, {f.result()[1]} errors")


def phase_rep0(items):
    rev = [it for it in items if it["both_orders"]]
    jobs = {c: list(tasks(c, items, 0)) for c in CONFIGS}
    jobs.update({f"{j}-pair-rev": list(tasks(f"{j}-pair", rev, 0, rev=True)) for j in {c.split("-")[0] for c in CONFIGS}})
    run_parallel(jobs)
    log("REP0_DONE")


def phase_retry(items):
    rev = [it for it in items if it["both_orders"]]
    jobs = {}
    for name, ts in [(c, tasks(c, items, 0)) for c in CONFIGS] + [(f"{j}-pair-rev", tasks(f"{j}-pair", rev, 0, True)) for j in {c.split("-")[0] for c in CONFIGS}]:
        todo = []
        for t in ts:
            p = t[4]
            if p.exists() and TRANSIENT.search(str(json.loads(p.read_text()).get("error", ""))):
                dst = FAILED / p.relative_to(CACHE)
                if dst.exists():  # already retried once; never twice
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(p, dst)
                todo.append(t)
        if todo:
            jobs[name] = todo
        log(f"{name}: {len(todo)} transient failures queued for the one retry")
    if jobs:
        run_parallel(jobs)
    log("RETRY_DONE")


def phase_noise(items):
    meta = json.loads((ROOT / "data" / "arena_pairs.json").read_text())["pairs"]
    D = Data(items, meta)
    V = {c: verdicts(c, items, 0) for c in CONFIGS}
    ok = eligible(V)
    for c in CONFIGS:
        v = V[c]; ran = v != -1
        log(f"{c}: coverage {ran.mean():.3f}, failures {(v[ran] == -2).mean():.3f}, eligible {c in ok}")
    sel_by = [D.by_pair[k][0] for k in range(len(D.pairs))]
    k = dict(zip(ok, kappa_from(per_pair_conf(D, ok, V, sel_by).sum(1)))) if ok else {}
    noise = [max((c for c in k if c.endswith(f)), key=k.get) for f in ("-pair", "-point") if any(c.endswith(f) for c in k)]
    log(f"NOISE_CONFIGS {noise}")  # selection-half kappas are not logged: they feed only this choice
    meas = [it for it in items if it["half"] == "measure"]
    for rep in (1, 2):
        run_parallel({c: list(tasks(c, meas, rep)) for c in noise})
        log(f"REP{rep}_DONE")
    log("DONE")


def main():
    global PACED
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judges", default=",".join(JUDGES), help="subset of judges (rep0/retry only)")
    a = ap.parse_args()
    global CONFIGS
    CONFIGS = [c for c in CONFIGS if c.split("-")[0] in a.judges.split(",")]
    STATUS.parent.mkdir(exist_ok=True)
    PACED = Paced(rj.client("nim"))
    items = load_items(a.limit)
    for name, fn in (("rep0", phase_rep0), ("retry", phase_retry), ("noise", phase_noise)):
        if a.phase in ("all", name):
            log(f"phase {name} start ({len(items)} items)")
            fn(items)


if __name__ == "__main__":
    main()
