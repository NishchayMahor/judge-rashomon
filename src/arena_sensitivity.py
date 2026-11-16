"""Secondary and sensitivity paths of PREREG-arena.md (section "Secondary and sensitivity"; all exploratory).

Every path reuses judge_rashomon.core. A path that changes one step swaps that one core function for its run, so all
other steps stay exactly the frozen primary path. Verdicts are in core's canonical encoding (see core docstring).
run() returns one dict per path; multiverse_figure() draws them in one labelled figure.
"""
import contextlib, itertools, re

import numpy as np

from judge_rashomon import core
from arena_analyze import JUDGE_FAMILIES, analyze

MD = re.compile(r"^\s*(#{1,6} |[-*+] |\d+\. |\|)|\*\*|```", re.M)


@contextlib.contextmanager
def swap(name, fn):
    old = getattr(core, name)
    setattr(core, name, fn)
    try:
        yield
    finally:
        setattr(core, name, old)


def families(V):
    return {c: JUDGE_FAMILIES[c.split("-")[0]] for c in V}


def summary(m):
    keys = ("eligible", "best", "R", "n_units", "flip", "perm_p", "flip_all", "slope", "slope_perm_p", "H1", "H2", "H3")
    out = {k: m[k] for k in keys if k in m}
    out["ci"] = {k: m["ci"][k] for k in ("flip", "flip_all")} if "ci" in m else {}
    return out


# ---- swapped core steps ----
def kappa_ties_dropped(conf):
    """Kappa over decisive judge verdicts only (primary: a judge tie counts as a disagreement)."""
    conf = conf.copy()
    conf[..., 2] = 0
    return KAPPA(conf)


KAPPA = core.kappa_from


def conclusions_ties_dropped(D, cfgs, V, meas_idx_by_pair):
    """Win rate over decisive judge verdicts only. sign((a + t/2)/(a + b + t) - 0.5) = sign(a - b), so this alone
    never changes a conclusion; the path differs from the primary one only through kappa_ties_dropped."""
    M = np.stack([V[c] for c in cfgs])
    out = np.zeros((len(cfgs), len(meas_idx_by_pair)))
    for j, idx in enumerate(meas_idx_by_pair):
        idx = idx[(M[:, idx] >= 0).all(0)]
        a, b = (M[:, idx] == 0).sum(1), (M[:, idx] == 1).sum(1)
        out[:, j] = np.where(a + b > 0, np.sign(a - b), 0)
    return out


def bt_strengths(first, second, w_first, iters=200):
    """Bradley-Terry by Zermelo/MM; w_first = 1, 0.5 (tie) or 0 per battle. A 0.01 pseudo-battle per observed pair
    keeps the fit finite when one side wins every battle."""
    models = sorted(set(first) | set(second))
    ix = {m: i for i, m in enumerate(models)}
    n = len(models)
    W, N = np.zeros((n, n)), np.zeros((n, n))
    for a, b, w in zip(first, second, w_first):
        i, j = ix[a], ix[b]
        W[i, j] += w; W[j, i] += 1 - w; N[i, j] += 1; N[j, i] += 1
    seen = N > 0
    W, N = W + 0.01 * seen, N + 0.02 * seen
    p = np.ones(n)
    for _ in range(iters):
        p = W.sum(1) / np.where(seen, N / (p[:, None] + p[None, :]), 0).sum(1)
        p /= np.exp(np.log(p).mean())
    return {m: float(np.log(p[ix[m]])) for m in models}


def conclusions_bt(D, cfgs, V, meas_idx_by_pair):
    """Per config: one BT fit over all measurement battles (complete-case within each pair); the conclusion for a pair
    is the sign of the strength difference instead of its own head-to-head win rate."""
    M = np.stack([V[c] for c in cfgs])
    kept = [idx[(M[:, idx] >= 0).all(0)] for idx in meas_idx_by_pair]
    allidx = np.concatenate(kept) if kept else np.array([], int)
    out = np.zeros((len(cfgs), len(kept)))
    if not len(allidx):
        return out
    fst = [D.pairs[D.pair_of[i]][0] for i in allidx]
    snd = [D.pairs[D.pair_of[i]][1] for i in allidx]
    for ci in range(len(cfgs)):
        v = M[ci, allidx]
        s = bt_strengths(fst, snd, np.where(v == 0, 1.0, np.where(v == 2, 0.5, 0.0)))
        for j, idx in enumerate(kept):
            if len(idx):
                a, b = D.pairs[D.pair_of[idx[0]]]
                out[ci, j] = np.sign(s[a] - s[b])
    return out


def select_fixed_eps(eps):
    def select(cfgs, Cp):
        k = core.kappa_from(Cp.sum(1))
        b = int(np.argmax(k))
        R = [cfgs[i] for i in range(len(cfgs)) if k[b] - k[i] <= eps]
        return cfgs[b], R, dict(zip(cfgs, k)), {c: float("nan") for c in cfgs}
    return select


# ---- paths ----
def subset(items, V, keep):
    keep = np.asarray(keep)
    return [it for it, k in zip(items, keep) if k], {c: v[keep] for c, v in V.items()}


def ten_word(items, meta10, V, reps, B, P):
    """Pre-registered rule: prompts with >= 10 words, human margin from >= 10-word votes, pairs with >= 60 such votes."""
    ok = {k for k, v in meta10.items() if v["n_nontie"] >= 60}
    keep = [len(it["prompt"].split()) >= 10 and "|".join(it["pair"]) in ok for it in items]
    its, Vs = subset(items, V, keep)
    rs = {c: [r[np.asarray(keep)] for r in rr] for c, rr in (reps or {}).items()} or None
    m = analyze(its, meta10, Vs, rs, B, P)
    return {**summary(m), "n_items": len(its), "n_pairs": len({tuple(it["pair"]) for it in its})}


def length_model(items, meta, V, R):
    """Flip ~ margin + log length ratio (A/B words) + markdown-density difference, per (config pair in R, model pair).
    Covariates are per model pair, averaged over its measurement items. Returns coefficients with Wald SEs."""
    core.FAMILIES.clear(); core.FAMILIES.update(families(V))
    D = core.Data(items, meta)
    ids = list(range(len(D.pairs)))
    meas_by = [D.by_pair[k][1] for k in ids]
    S = core.conclusions(D, R, V, meas_by)
    fl, j = core.flip_units(D, R, S, ids, R, np.arange(len(ids), dtype=float))  # margin slot carries the pair index
    j = j.astype(int)
    words = lambda s: len(s.split()) + 1
    mdd = lambda s: 100 * len(MD.findall(s)) / words(s)
    lr = np.array([np.mean([np.log(words(items[i]["ans_first"]) / words(items[i]["ans_second"])) for i in meas_by[k]]) for k in ids])
    md = np.array([np.mean([mdd(items[i]["ans_first"]) - mdd(items[i]["ans_second"]) for i in meas_by[k]]) for k in ids])
    m = np.abs(D.p_first - 0.5)
    X = np.column_stack([np.ones(len(j)), m[j], lr[j], md[j]])
    if len(fl) < 8 or fl.min() == fl.max():
        return {"n_units": len(fl), "note": "undefined (too few units, or no variation in flips)"}
    w = np.zeros(4)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(4)
        w = w + np.linalg.solve(H, X.T @ (fl - p))
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    names = ("intercept", "margin", "log_len_ratio", "markdown_diff")
    return {"n_units": len(fl), "coef": dict(zip(names, w.tolist())), "se": dict(zip(names, se.tolist()))}


def order_agreement(items, V, Vrev):
    """Both-orders subset: share of items where a pairwise config gives the same canonical verdict in both orders."""
    sub = np.array([it["both_orders"] for it in items])
    out = {}
    for c, vr in Vrev.items():
        a, b = V[c][sub], vr[sub]
        ok = (a >= 0) & (b >= 0)
        out[c] = {"n": int(ok.sum()), "agree": float((a[ok] == b[ok]).mean()) if ok.any() else float("nan")}
    return out


def ambiguity(items, meta, V, R, best, reps):
    """Marx et al. ambiguity on the measurement half: share of complete-case items where some config in R differs from
    the best config, against the same share between the best config's rep 0 and its re-runs (when available)."""
    D = core.Data(items, meta)
    meas = ~D.sel
    M = np.stack([V[c] for c in R])
    ok = meas & (M >= 0).all(0)
    out = {"in_R": float((M[:, ok] != V[best][ok]).any(0).mean()) if ok.any() else float("nan"), "n_items": int(ok.sum())}
    if reps and best in reps:
        out["rerun_floor"] = [float((r[ok & (r >= 0)] != V[best][ok & (r >= 0)]).mean()) for r in reps[best]]
    return out


def classes_and_confident(items, meta, V, R, B=200, seed=0):
    """Descriptive human-closeness classes and confident flips (both configs' bootstrap win-rate CIs exclude 0.5 with
    opposite signs), per scored (config pair in R, model pair) unit."""
    core.FAMILIES.clear(); core.FAMILIES.update(families(V))
    D = core.Data(items, meta)
    rng = np.random.default_rng(seed)
    n, p = D.n_votes, D.p_first
    half = 1.96 * np.sqrt(p * (1 - p) / n)
    lo, hi = p - half, p + half
    cls = np.where((np.abs(p - 0.5) > 0.1) & ((lo > 0.5) | (hi < 0.5)), "clear",
                   np.where((lo >= 0.4) & (hi <= 0.6), "contested", "unresolved"))
    ids = list(range(len(D.pairs)))
    meas_by = [D.by_pair[k][1] for k in ids]
    S = core.conclusions(D, R, V, meas_by)
    fl, j = core.flip_units(D, R, S, ids, R, np.arange(len(ids), dtype=float))
    j = j.astype(int)
    M = np.stack([V[c] for c in R])
    conf = np.zeros((len(R), len(ids)))  # +1 / -1 when the win-rate CI excludes 0.5, else 0
    for k, idx in enumerate(meas_by):
        idx = idx[(M[:, idx] >= 0).all(0)]
        if not len(idx):
            continue
        w = (M[:, idx] == 0) + 0.5 * (M[:, idx] == 2)
        bs = np.stack([w[:, rng.integers(0, len(idx), len(idx))].mean(1) for _ in range(B)])
        l, h = np.percentile(bs, [2.5, 97.5], axis=0)
        conf[:, k] = np.where(l > 0.5, 1, np.where(h < 0.5, -1, 0))
    conf_flips = 0
    for a, b in itertools.combinations(range(len(R)), 2):
        for k in ids:
            if core.family_clash(R[a], D.pairs[k]) or core.family_clash(R[b], D.pairs[k]):
                continue
            conf_flips += int(conf[a, k] * conf[b, k] == -1)
    by_class = {c: {"pairs": int((cls == c).sum()),
                    "units": int((cls[j] == c).sum()),
                    "flip": float(fl[cls[j] == c].mean()) if (cls[j] == c).any() else float("nan")}
                for c in ("clear", "contested", "unresolved")}
    return {"classes": by_class, "confident_flips": conf_flips, "n_units": len(fl)}


def run(items, meta, V, reps=None, Vrev=None, meta10=None, B=2000, P=1000):
    res = {}
    prim = analyze(items, meta, V, reps, B, P)
    res["primary"] = summary(prim)
    R, best = prim["R"], prim["best"]
    if meta10:
        res["ten_word"] = ten_word(items, meta10, V, reps, B, P)
    res["no_family"] = summary(core.analyze(items, meta, V, reps, B, P, families={}))
    with swap("conclusions", conclusions_ties_dropped), swap("kappa_from", kappa_ties_dropped):
        res["ties_dropped"] = summary(analyze(items, meta, V, reps, B, P))
    with swap("conclusions", conclusions_bt):
        res["bradley_terry"] = summary(analyze(items, meta, V, reps, B, P))
    grid = []
    core.FAMILIES.clear(); core.FAMILIES.update(families(V))
    D = core.Data(items, meta)
    ids = list(range(len(D.pairs)))
    cfgs = core.eligible(V)
    for eps in np.round(np.arange(0, 0.201, 0.02), 2):
        with swap("select_set", select_fixed_eps(eps)):
            m = core.one_pass(D, cfgs, V, [D.by_pair[k][0] for k in ids], [D.by_pair[k][1] for k in ids], ids,
                              np.abs(D.p_first - 0.5))[0]
        grid.append({"eps": float(eps), "R_size": len(m["R"]), "flip": m["flip"]})
    res["eps_grid"] = grid
    res["length"] = length_model(items, meta, V, R) if len(R) > 1 else {"note": "R < 2"}
    if Vrev:
        res["order"] = order_agreement(items, V, Vrev)
    res["ambiguity"] = ambiguity(items, meta, V, R, best, reps)
    res["classes"] = classes_and_confident(items, meta, V, R) if len(R) > 1 else {"note": "R < 2"}
    return res


def multiverse_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from judge_rashomon.figures import ACCENT, INK, MUTED, _style
    paths = [k for k in ("primary", "ten_word", "no_family", "ties_dropped", "bradley_terry") if k in res]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
    for i, k in enumerate(paths):
        r = res[k]
        f, ci = r.get("flip", float("nan")), r.get("ci", {}).get("flip", [float("nan")] * 2)
        col = ACCENT if k == "primary" else INK
        if f == f:
            a1.errorbar(f, i, xerr=[[f - ci[0]], [ci[1] - f]] if ci[0] == ci[0] else None, fmt="o", color=col, capsize=3)
        a1.annotate(f"|R|={len(r.get('R', []))}", (0, i), xytext=(4, 5), textcoords="offset points", fontsize=7, color=MUTED)
    a1.set_yticks(range(len(paths)), [p.replace("_", " ") for p in paths]); a1.invert_yaxis()
    _style(a1, "Flip rate inside the set, by analysis path", "flip rate (95% CI)", "")
    g = res["eps_grid"]
    a2.plot([x["eps"] for x in g], [x["flip"] for x in g], "o-", color=ACCENT)
    for x in g:
        a2.annotate(str(x["R_size"]), (x["eps"], x["flip"]), xytext=(0, 5), textcoords="offset points", fontsize=7,
                    ha="center", color=MUTED)
    _style(a2, "Fixed-eps sets (labels = set size)", "eps (kappa gap)", "flip rate")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
