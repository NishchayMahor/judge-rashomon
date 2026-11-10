"""Compute the pre-registered pilot metrics from cache/. Writes results/pilot_metrics.json.

Usage: uv run python src/analyze.py [--cache cache] [--boot 1000]
"""
import argparse, collections, hashlib, itertools, json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score

from run_judges import CONFIGS, parse_pair, parse_point

ROOT = Path(__file__).resolve().parent.parent
EPS = [0.02, 0.05, 0.10]
LABELS = ["model_a", "model_b", "tie"]


def load_verdicts(cache, cfg, rep, items):
    """Verdict per item in canonical (model_a, model_b) orientation; None if missing/unparsed."""
    mkey, fmt = cfg.split("-")
    out = []
    for it in items:
        if fmt == "pair":
            p = cache / f"rep{rep}" / cfg / f"{it['id']}.json"
            v = parse_pair(json.loads(p.read_text())["text"]) if p.exists() else None
            if v and it["swap"] and v != "tie":
                v = "model_b" if v == "model_a" else "model_a"
        else:
            s = []
            for side in ("a", "b"):
                h = hashlib.sha1((it["question"] + it[f"ans_{side}"]).encode()).hexdigest()[:12]
                p = cache / f"rep{rep}" / f"{mkey}-point" / f"{it['qid']}_{h}.json"
                s.append(parse_point(json.loads(p.read_text())["text"]) if p.exists() else None)
            v = None if None in s else ("model_a" if s[0] > s[1] else "model_b" if s[1] > s[0] else "tie")
        out.append(v)
    return out


def system_scores(items, verdicts, idx):
    win, n = collections.Counter(), collections.Counter()
    for i in idx:
        it, v = items[i], verdicts[i] or "tie"
        a, b = it["model_a"], it["model_b"]
        win[a] += {"model_a": 1, "tie": 0.5}.get(v, 0)
        win[b] += {"model_b": 1, "tie": 0.5}.get(v, 0)
        n[a] += 1; n[b] += 1
    return {m: win[m] / n[m] for m in n}


def flip_rate(s1, s2):
    flips = tot = 0
    for m1, m2 in itertools.combinations(sorted(s1), 2):
        d1, d2 = np.sign(s1[m1] - s1[m2]), np.sign(s2[m1] - s2[m2])
        if d1 == 0 or d2 == 0:
            continue
        tot += 1; flips += d1 != d2
    return flips / tot if tot else float("nan")


def tau(s1, s2):
    ms = sorted(s1)
    return kendalltau([s1[m] for m in ms], [s2[m] for m in ms]).statistic


def kappa(items, v, idx):
    return cohen_kappa_score([items[i]["human"] for i in idx], [v[i] or "tie" for i in idx], labels=LABELS)


def rashomon_stats(items, V, idx, eps):
    ks = {c: kappa(items, V[c], idx) for c in V}
    best = max(ks, key=ks.get)
    R = [c for c in V if ks[c] >= ks[best] - eps]
    S = {c: system_scores(items, V[c], idx) for c in R}
    pairs = list(itertools.combinations(R, 2))
    fr = np.nanmean([flip_rate(S[a], S[b]) for a, b in pairs]) if pairs else float("nan")
    tk = np.nanmean([tau(S[a], S[b]) for a, b in pairs]) if pairs else float("nan")
    amb = np.mean([any((V[c][i] or "tie") != (V[best][i] or "tie") for c in R) for i in idx])
    return dict(best=best, R=R, size=len(R), flip_rate=fr, tau=tk, ambiguity=amb)


def noise_stats(items, reps, idx):
    S = [system_scores(items, v, idx) for v in reps]
    pr = list(itertools.combinations(range(len(reps)), 2))
    return dict(flip_rate=np.nanmean([flip_rate(S[a], S[b]) for a, b in pr]),
                tau=np.nanmean([tau(S[a], S[b]) for a, b in pr]),
                ambiguity=np.mean([any((r[i] or "tie") != (reps[0][i] or "tie") for r in reps) for i in idx]))


def human_ceiling():
    from datasets import load_dataset
    h = load_dataset("lmsys/mt_bench_human_judgments")["human"]
    g = collections.defaultdict(list)
    for r in h:
        if str(r["turn"]) != "1":
            continue
        a, b, w = r["model_a"], r["model_b"], r["winner"]
        if a > b:
            a, b, w = b, a, {"model_a": "model_b", "model_b": "model_a"}.get(w, w)
        g[(r["question_id"], a, b)].append(w)
    agree = [x == y for v in g.values() if len(v) > 1 for x, y in itertools.combinations(v, 2)]
    return dict(n_items_multi=sum(len(v) > 1 for v in g.values()), pairwise_agreement=float(np.mean(agree)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--noise", default="")
    ap.add_argument("--no-cli", action="store_true")
    ap.add_argument("--out", default="pilot_metrics.json")  # comma list of configs with rep1, rep2
    args = ap.parse_args()
    cache = Path(args.cache)
    items = json.loads((ROOT / "data" / "pilot_items.json").read_text())
    # coverage: an item counts as covered when its cache file(s) exist (an unparsed reply is still covered)
    def covered(c, rep=0):
        mkey, fmt = c.split("-")
        if fmt == "pair":
            return [(cache / f"rep{rep}" / c / f"{it['id']}.json").exists() for it in items]
        return [all((cache / f"rep{rep}" / c / f"{it['qid']}_{hashlib.sha1((it['question'] + it[f'ans_{s}']).encode()).hexdigest()[:12]}.json").exists()
                    for s in ("a", "b")) for it in items]
    cov = {c: covered(c) for c in CONFIGS if (cache / "rep0" / c).exists()}
    full = [c for c in cov if np.mean(cov[c]) >= 0.9]
    excluded = {c: float(np.mean(cov[c])) for c in cov if c not in full}
    if args.no_cli:
        full = [c for c in full if c.split("-")[0] not in ("haiku", "sonnet")]
    idx = np.array([i for i in range(len(items)) if all(cov[c][i] for c in full)])
    V = {c: load_verdicts(cache, c, 0, items) for c in full}
    res = {"n_items": int(len(idx)), "excluded_low_coverage": excluded, "configs": {}}
    for c, v in V.items():
        res["configs"][c] = dict(kappa=kappa(items, v, idx), parse_fail=sum(x is None for x in v),
                                 tie_rate=float(np.mean([x == "tie" for x in v])),
                                 scores=system_scores(items, v, idx))
    rng = np.random.default_rng(0)
    boots = [rng.choice(idx, len(idx)) for _ in range(args.boot)]
    res["rashomon"] = {}
    for e in EPS:
        pt = rashomon_stats(items, V, idx, e)
        bs = [rashomon_stats(items, V, b, e) for b in boots]
        res["rashomon"][str(e)] = {**pt, **{f"{k}_se": float(np.nanstd([b[k] for b in bs])) for k in ("flip_rate", "tau", "ambiguity", "size")}}
    res["noise"] = {}
    for c in filter(None, args.noise.split(",")):
        nidx = np.array([i for i in idx if all(covered(c, r)[i] for r in (1, 2))])
        reps = [load_verdicts(cache, c, r, items) for r in (0, 1, 2)]
        pt = noise_stats(items, reps, nidx)
        nb = [rng.choice(nidx, len(nidx)) for _ in range(args.boot)]
        bs = [noise_stats(items, reps, b) for b in nb]
        res["noise"][c] = {**pt, **{f"{k}_se": float(np.nanstd([b[k] for b in bs])) for k in pt}}
    res["human_ceiling"] = human_ceiling()
    out = ROOT / "results" / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("rashomon", "noise", "human_ceiling")}, indent=1, default=float))
    for c, r in sorted(res["configs"].items(), key=lambda x: -x[1]["kappa"]):
        print(f"{c:15s} kappa={r['kappa']:.3f} fail={r['parse_fail']} tie={r['tie_rate']:.2f}")


if __name__ == "__main__":
    main()
