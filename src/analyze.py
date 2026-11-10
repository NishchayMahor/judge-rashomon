"""Pilot metrics per PREREG (with 2026-11-10 amendment A1-A10). Writes results/<out>.

Primary path: complete-case verdicts, configs with <=5% failures, CLI configs excluded, question-cluster bootstrap,
epsilon-Rashomon set on paired kappa, confident flips, verdict-swap permutation null, set-vs-all contrast,
2-fold selection/measurement split (dry run in the pilot).
Usage: uv run python src/analyze.py [--noise c1,c2] [--with-cli] [--ties-as-tie] [--out pilot_metrics.json]
"""
import argparse, collections, hashlib, itertools, json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

from run_judges import CONFIGS, parse_pair, parse_point

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["model_a", "model_b", "tie"]
EPS_GRID = [round(x, 2) for x in np.arange(0, 0.21, 0.01)]
EPS_PRIMARY = 0.05
CLI = ("haiku", "sonnet")
MAX_FAIL = 0.05
CLEAR_GAP = 0.15  # |human win rate - 0.5| at or above this = clear pair


def cache_files(cache, cfg, rep, it):
    mkey, fmt = cfg.split("-")
    if fmt == "pair":
        return [cache / f"rep{rep}" / cfg / f"{it['id']}.json"]
    return [cache / f"rep{rep}" / cfg / f"{it['qid']}_{hashlib.sha1((it['question'] + it[f'ans_{s}']).encode()).hexdigest()[:12]}.json"
            for s in ("a", "b")]


def load_verdicts(cache, cfg, rep, items):
    """Canonical-orientation verdict per item: label, 'FAIL' (call made, unusable) or None (not run)."""
    out = []
    for it in items:
        fs = cache_files(cache, cfg, rep, it)
        if not all(f.exists() for f in fs):
            out.append(None)
            continue
        texts = [json.loads(f.read_text()).get("text", "") for f in fs]
        if cfg.endswith("-pair"):
            v = parse_pair(texts[0])
            if v and it["swap"] and v != "tie":
                v = "model_b" if v == "model_a" else "model_a"
        else:
            s = [parse_point(t) for t in texts]
            if any(x is None or not 1 <= x <= 10 for x in s):
                v = None
            else:
                v = "model_a" if s[0] > s[1] else "model_b" if s[1] > s[0] else "tie"
        out.append(v or "FAIL")
    return out


def system_scores(items, v, idx):
    win, n = collections.Counter(), collections.Counter()
    for i in idx:
        it = items[i]
        a, b = it["model_a"], it["model_b"]
        win[a] += {"model_a": 1, "tie": 0.5}.get(v[i], 0)
        win[b] += {"model_b": 1, "tie": 0.5}.get(v[i], 0)
        n[a] += 1; n[b] += 1
    return {m: win[m] / n[m] for m in n}


def pair_diffs(s):
    return {(m1, m2): s[m1] - s[m2] for m1, m2 in itertools.combinations(sorted(s), 2)}


def flip_rate(d1, d2, pairs=None):
    flips = tot = 0
    for p in (pairs or d1):
        a, b = np.sign(d1[p]), np.sign(d2[p])
        if a == 0 or b == 0:
            continue
        tot += 1; flips += a != b
    return flips / tot if tot else float("nan")


def kendall(d1, d2):
    c = [np.sign(d1[p]) * np.sign(d2[p]) for p in d1]
    return float(np.mean(c))  # tau-a over model pairs


def kappa(items, v, idx):
    return cohen_kappa_score([items[i]["human"] for i in idx], [v[i] for i in idx], labels=LABELS)


def cluster_boot(items, idx, rng, B):
    by_q = collections.defaultdict(list)
    for i in idx:
        by_q[items[i]["qid"]].append(i)
    qs = list(by_q)
    return [np.concatenate([by_q[q] for q in rng.choice(qs, len(qs))]) for _ in range(B)]


def rashomon(ks, eps):
    best = max(ks, key=ks.get)
    return best, [c for c in ks if ks[c] >= ks[best] - eps]


def mean_flip(D, R, pairs=None):
    pr = list(itertools.combinations(R, 2))
    return float(np.nanmean([flip_rate(D[a], D[b], pairs) for a, b in pr])) if pr else float("nan")


def ambiguity(V, R, best, idx):
    return float(np.mean([any(V[c][i] != V[best][i] for c in R) for i in idx]))


def confident_flips(items, V, R, idx, boots):
    """Share of (config pair, model pair) where both configs' 95% CIs exclude 0 with opposite signs."""
    ci = {}
    for c in R:
        bd = [pair_diffs(system_scores(items, V[c], b)) for b in boots]
        ci[c] = {p: np.percentile([d[p] for d in bd], [2.5, 97.5]) for p in bd[0]}
    sig = lambda c, p: 1 if ci[c][p][0] > 0 else -1 if ci[c][p][1] < 0 else 0
    tot = flips = 0
    for a, b in itertools.combinations(R, 2):
        for p in ci[a]:
            tot += 1; flips += sig(a, p) * sig(b, p) == -1
    return flips / tot if tot else float("nan")


def perm_null(items, V, R, idx, rng, P):
    """Verdict-swap permutation: for each config pair, swap verdicts item-wise with p=0.5."""
    pr = list(itertools.combinations(R, 2))
    if not pr:
        return []
    null = []
    for _ in range(P):
        fr = []
        for a, b in pr:
            m = rng.random(len(items)) < 0.5
            va = [V[b][i] if m[i] else V[a][i] for i in range(len(items))]
            vb = [V[a][i] if m[i] else V[b][i] for i in range(len(items))]
            fr.append(flip_rate(pair_diffs(system_scores(items, va, idx)), pair_diffs(system_scores(items, vb, idx))))
        null.append(float(np.nanmean(fr)))
    return null


def human_pair_gaps(items, idx):
    hv = {i: items[i]["human"] for i in idx}
    d = pair_diffs(system_scores(items, hv, idx))
    return {p: abs(x) / 2 for p, x in d.items()}  # diff of win rates / 2 ~ |w - 0.5| on the pair


def analyze_block(items, V, idx, rng, B, P):
    ks = {c: kappa(items, V[c], idx) for c in V}
    D = {c: pair_diffs(system_scores(items, V[c], idx)) for c in V}
    boots = cluster_boot(items, idx, rng, B)
    gaps = human_pair_gaps(items, idx)
    clear = [p for p, g in gaps.items() if g >= CLEAR_GAP]
    contested = [p for p in gaps if p not in clear]
    curve = {}
    for e in EPS_GRID:
        best, R = rashomon(ks, e)
        curve[str(e)] = dict(size=len(R), flip=mean_flip(D, R), flip_clear=mean_flip(D, R, clear),
                             flip_contested=mean_flip(D, R, contested))
    best, R = rashomon(ks, EPS_PRIMARY)
    bk = [{c: kappa(items, V[c], b) for c in V} for b in boots]
    member = {c: float(np.mean([c in rashomon(k, EPS_PRIMARY)[1] for k in bk])) for c in V}
    paired_se = {c: float(np.std([k[best] - k[c] for k in bk])) for c in V if c != best}
    bf = [mean_flip({c: pair_diffs(system_scores(items, V[c], b)) for c in R}, R) for b in boots[:200]]
    null = perm_null(items, V, R, idx, rng, P)
    obs = mean_flip(D, R)
    allc = list(V)
    return dict(
        n_items=int(len(idx)), kappa={c: float(k) for c, k in ks.items()},
        tie_rate={c: float(np.mean([V[c][i] == "tie" for i in idx])) for c in V},
        best=best, R=R, paired_kappa_se=paired_se, membership_freq=member,
        flip=obs, flip_se=float(np.nanstd(bf)), confident_flip=confident_flips(items, V, R, idx, boots[:300]),
        ambiguity=ambiguity(V, R, best, idx), tau_mean=float(np.mean([kendall(D[a], D[b]) for a, b in itertools.combinations(R, 2)])) if len(R) > 1 else None,
        perm_null_mean=float(np.mean(null)) if null else None, perm_p=float(np.mean([x >= obs for x in null])) if null else None,
        flip_all_configs=mean_flip(D, allc), confident_flip_all=confident_flips(items, V, allc, idx, boots[:300]),
        n_clear_pairs=len(clear), n_contested_pairs=len(contested), eps_curve=curve)


def split_half(items, V, idx, rng, folds=2):
    """Dry run of A1: select best/R on one fold of questions, measure flips on the other."""
    qs = sorted({items[i]["qid"] for i in idx})
    rng.shuffle(qs)
    half = set(qs[: len(qs) // 2])
    F = [np.array([i for i in idx if items[i]["qid"] in half]), np.array([i for i in idx if items[i]["qid"] not in half])]
    out = []
    for s, m in ((0, 1), (1, 0)):
        ks = {c: kappa(items, V[c], F[s]) for c in V}
        best, R = rashomon(ks, EPS_PRIMARY)
        D = {c: pair_diffs(system_scores(items, V[c], F[m])) for c in V}
        km = {c: float(kappa(items, V[c], F[m])) for c in V}
        out.append(dict(select_best=best, R=R, flip_measured=mean_flip(D, R), flip_all=mean_flip(D, list(V)),
                        best_kappa_select=float(ks[best]), best_kappa_measure=km[best]))
    return out


def noise_stats(items, reps, idx):
    D = [pair_diffs(system_scores(items, v, idx)) for v in reps]
    pr = list(itertools.combinations(range(len(reps)), 2))
    return dict(flip=float(np.nanmean([flip_rate(D[a], D[b]) for a, b in pr])),
                verdict_change=float(np.mean([any(r[i] != reps[0][i] for r in reps) for i in idx])), n_items=int(len(idx)))


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
    ap.add_argument("--perm", type=int, default=500)
    ap.add_argument("--noise", default="")
    ap.add_argument("--with-cli", action="store_true")
    ap.add_argument("--ties-as-tie", action="store_true", help="sensitivity: failures scored as ties")
    ap.add_argument("--out", default="pilot_metrics.json")
    args = ap.parse_args()
    cache = Path(args.cache)
    items = json.loads((ROOT / "data" / "pilot_items.json").read_text())
    raw = {c: load_verdicts(cache, c, 0, items) for c in CONFIGS if (cache / "rep0" / c).exists()}
    status = {c: dict(coverage=float(np.mean([x is not None for x in v])),
                      fail_rate=float(np.mean([x == "FAIL" for x in v if x is not None])) if any(x is not None for x in v) else None)
              for c, v in raw.items()}
    keep = [c for c, s in status.items() if s["coverage"] >= 0.9 and s["fail_rate"] <= MAX_FAIL
            and (args.with_cli or c.split("-")[0] not in CLI)]
    if args.ties_as_tie:
        raw = {c: [("tie" if x == "FAIL" else x) for x in v] for c, v in raw.items()}
    V = {c: raw[c] for c in keep}
    idx = np.array([i for i in range(len(items)) if all(V[c][i] not in (None, "FAIL") for c in V)])
    rng = np.random.default_rng(0)
    res = dict(path=dict(with_cli=args.with_cli, ties_as_tie=args.ties_as_tie, eps_primary=EPS_PRIMARY, max_fail=MAX_FAIL),
               config_status=status, included=keep, main=analyze_block(items, V, idx, rng, args.boot, args.perm),
               split_half=split_half(items, V, idx, rng), human_ceiling=human_ceiling(), noise={})
    for c in filter(None, args.noise.split(",")):
        reps = [load_verdicts(cache, c, r, items) for r in (0, 1, 2)]
        nidx = np.array([i for i in range(len(items)) if all(r[i] not in (None, "FAIL") for r in reps)])
        if len(nidx):
            res["noise"][c] = noise_stats(items, reps, nidx)
    out = ROOT / "results" / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    m = res["main"]
    print(f"items={m['n_items']} included={keep}")
    print(f"best={m['best']} R={m['R']} flip={m['flip']:.3f}+-{m['flip_se']:.3f} conf={m['confident_flip']:.3f} "
          f"perm_null={m['perm_null_mean']} p={m['perm_p']} all={m['flip_all_configs']:.3f}")


if __name__ == "__main__":
    main()
