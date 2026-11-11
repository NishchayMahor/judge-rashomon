"""Core of judge-rashomon: select the Rashomon set of judge configs on a selection split, measure flips of system-level
conclusions on a measurement split, and test them against a joint permutation null and a re-run noise floor.

Verdicts are int arrays per config over the items, in canonical (sorted-pair) orientation:
0 = first system better, 1 = second better, 2 = tie, -1 = not run, -2 = failed. Humans are 0/1 (ties dropped).
families maps a config to contestant-name prefixes it must not judge (self-preference rule); configs not in it judge all.
"""
import itertools

import numpy as np

MAX_FAIL, MIN_COV = 0.05, 0.90
FAMILIES = {}  # config -> prefixes; set per call by analyze()


def family_clash(cfg, pair):
    return any(m.startswith(tuple(FAMILIES.get(cfg.split("#")[0], ()))) for m in pair)


class Data:
    def __init__(self, items, pairs_meta):
        self.pairs = sorted({tuple(it["pair"]) for it in items})
        pid = {p: k for k, p in enumerate(self.pairs)}
        self.pair_of = np.array([pid[tuple(it["pair"])] for it in items])
        self.sel = np.array([it["half"] == "select" for it in items])
        self.h = np.array([0 if it["human"] == "first" else 1 for it in items])
        meta = [pairs_meta["|".join(p)] for p in self.pairs]
        self.n_votes = np.array([m["n_nontie"] for m in meta])
        self.p_first = np.array([m["p_first"] for m in meta])
        self.by_pair = [(np.where((self.pair_of == k) & self.sel)[0], np.where((self.pair_of == k) & ~self.sel)[0])
                        for k in range(len(self.pairs))]


def eligible(V):
    out = []
    for c, v in V.items():
        ran = v != -1
        if ran.mean() >= MIN_COV and (v[ran] == -2).mean() <= MAX_FAIL:
            out.append(c)
    return out


def kappa_from(conf):
    """conf[..., 2, 3]: human (first/second) x judge (first/second/tie) counts."""
    n = conf.sum((-1, -2))
    po = (conf[..., 0, 0] + conf[..., 1, 1]) / n
    hr, jc = conf.sum(-1), conf.sum(-2)
    pe = (hr[..., 0] * jc[..., 0] + hr[..., 1] * jc[..., 1]) / n ** 2
    return (po - pe) / (1 - pe)


def per_pair_conf(D, cfgs, V, sel_idx_by_pair):
    """[config, pair-instance, 2, 3] confusion over complete-case selection items."""
    out = np.zeros((len(cfgs), len(sel_idx_by_pair), 2, 3))
    M = np.stack([V[c] for c in cfgs])
    for j, idx in enumerate(sel_idx_by_pair):
        idx = idx[(M[:, idx] >= 0).all(0)]
        for ci in range(len(cfgs)):
            np.add.at(out[ci, j], (D.h[idx], M[ci, idx]), 1)
    return out


def select_set(cfgs, Cp):
    """Best by kappa; R = configs whose paired kappa gap to the best is within 1 cluster-jackknife SE."""
    k = kappa_from(Cp.sum(1))
    b = int(np.argmax(k))
    J = Cp.shape[1]
    tot = Cp.sum(1)
    kj = kappa_from(tot[:, None] - Cp)  # [config, left-out instance]
    gap = k[b] - k
    gj = kj[b] - kj
    se = np.sqrt((J - 1) / J * ((gj - gj.mean(1, keepdims=True)) ** 2).sum(1))
    R = [cfgs[i] for i in range(len(cfgs)) if i == b or gap[i] <= se[i]]
    return cfgs[b], R, dict(zip(cfgs, k)), dict(zip(cfgs, se))


def conclusions(D, cfgs, V, meas_idx_by_pair):
    """sign(win rate of first - 0.5) per config per pair-instance on complete-case measurement items; 0 = none."""
    M = np.stack([V[c] for c in cfgs])
    out = np.zeros((len(cfgs), len(meas_idx_by_pair)))
    for j, idx in enumerate(meas_idx_by_pair):
        idx = idx[(M[:, idx] >= 0).all(0)]
        if len(idx):
            w = ((M[:, idx] == 0) + 0.5 * (M[:, idx] == 2)).mean(1)
            out[:, j] = np.sign(w - 0.5)
    return out


def flip_units(D, cfgs, S, pair_ids, R, margins):
    """(flip, margin) for each scored (config pair in R, model pair) unit, applying the family rule."""
    ci = {c: i for i, c in enumerate(cfgs)}
    fl, mg = [], []
    for a, b in itertools.combinations(R, 2):
        for j, p in enumerate(pair_ids):
            pair = D.pairs[p]
            if family_clash(a, pair) or family_clash(b, pair):
                continue
            sa, sb = S[ci[a], j], S[ci[b], j]
            if sa == 0 or sb == 0:
                continue
            fl.append(float(sa != sb)); mg.append(margins[j])
    return np.array(fl), np.array(mg)


def logistic(y, x, iters=50):
    """Flip ~ margin by IRLS; returns (intercept, slope) or None when undefined (no flips or all flips)."""
    if len(y) < 3 or y.min() == y.max():
        return None
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(2)
        w = w + np.linalg.solve(H, X.T @ (y - p))
    return w


def pred(w, m):
    return float("nan") if w is None else float(1 / (1 + np.exp(-(w[0] + w[1] * m))))


def one_pass(D, cfgs, V, sel_by, meas_by, pair_ids, margins):
    best, R, ks, se = select_set(cfgs, per_pair_conf(D, cfgs, V, sel_by))
    S = conclusions(D, cfgs, V, meas_by)
    fl, mg = flip_units(D, cfgs, S, pair_ids, R, margins)
    fa, _ = flip_units(D, cfgs, S, pair_ids, cfgs, margins)
    w = logistic(fl, mg)
    return dict(best=best, R=R, kappa=ks, gap_se=se, n_units=len(fl), flip=float(fl.mean()) if len(fl) else float("nan"),
                flip_all=float(fa.mean()) if len(fa) else float("nan"), slope=float("nan") if w is None else float(w[1]),
                p_at_005=pred(w, 0.05), p_at_020=pred(w, 0.20)), S, R


def perm_null(D, cfgs, V, R, meas_by, pair_ids, margins, rng, P):
    """Joint verdict-permutation null on the measurement half: on every item, shuffle the verdicts of the configs in R
    among themselves. One shuffle per item for all of R keeps the correlation between config pairs that share a config
    (pairwise swaps break it and were anti-conservative in the synthetic dry run). R comes from the untouched
    selection half, so this re-runs every step that the measurement data feeds."""
    meas = np.concatenate(meas_by)
    M = np.stack([V[c] for c in R])
    out, slopes = [], []
    for _ in range(P):
        perm = np.argsort(rng.random((len(R), len(meas))), axis=0)
        Mp = M.copy()
        Mp[:, meas] = np.take_along_axis(M[:, meas], perm, axis=0)
        S = conclusions(D, R, dict(zip(R, Mp)), meas_by)
        fl, mg = flip_units(D, R, S, pair_ids, R, margins)
        out.append(float(np.mean(fl)) if len(fl) else float("nan"))
        w = logistic(fl, mg)
        slopes.append(float("nan") if w is None else float(w[1]))
    return np.array(out), np.array(slopes)


def analyze(items, pairs_meta, V, reps=None, B=2000, P=1000, seed=0, families=None):
    """V: {config: int array}. reps: {config: [rep1 array, rep2 array]} for the noise floor."""
    FAMILIES.clear(); FAMILIES.update(families or {})
    D = Data(items, pairs_meta)
    rng = np.random.default_rng(seed)
    cfgs = eligible(V)
    ids = list(range(len(D.pairs)))
    margins = np.abs(D.p_first - 0.5)
    sel_by = [D.by_pair[k][0] for k in ids]
    meas_by = [D.by_pair[k][1] for k in ids]
    main, S, R = one_pass(D, cfgs, V, sel_by, meas_by, ids, margins)
    null, null_slope = perm_null(D, cfgs, V, R, meas_by, ids, margins, rng, P) if len(R) > 1 else (np.array([]), np.array([]))
    main["perm_null_mean"] = float(np.nanmean(null)) if len(null) else float("nan")
    main["perm_p"] = float(np.mean(null >= main["flip"])) if len(null) else float("nan")
    # H3 is against the null too: sampling alone makes flips fall with the human margin
    main["perm_null_slope_mean"] = float(np.nanmean(null_slope)) if len(null_slope) else float("nan")
    main["slope_perm_p"] = float(np.nanmean(null_slope <= main["slope"])) if len(null_slope) else float("nan")
    boots = []
    for _ in range(B):  # cluster bootstrap: pairs, then battles within pair; human margin from resampled votes
        pk = rng.choice(ids, len(ids))
        sb = [rng.choice(D.by_pair[k][0], len(D.by_pair[k][0])) for k in pk]
        mb = [rng.choice(D.by_pair[k][1], len(D.by_pair[k][1])) for k in pk]
        mg = np.abs(rng.binomial(D.n_votes[pk], D.p_first[pk]) / D.n_votes[pk] - 0.5)
        boots.append(one_pass(D, cfgs, V, sb, mb, list(pk), mg)[0])
    ci = lambda k: [float(np.nanpercentile([b[k] for b in boots], q)) for q in (2.5, 97.5)]
    main["ci"] = {k: ci(k) for k in ("flip", "flip_all", "slope", "p_at_005", "p_at_020")}
    main["slope_upper_95_one_sided"] = float(np.nanpercentile([b["slope"] for b in boots], 95))
    main["contrast_all_minus_R"] = [float(x) for x in np.nanpercentile([b["flip_all"] - b["flip"] for b in boots], [2.5, 97.5])]
    main["membership_freq"] = {c: float(np.mean([c in b["R"] for b in boots])) for c in cfgs}
    if reps:  # noise floor: re-run vs rep0 of the same config on the same units
        nf = {}
        for c, rr in reps.items():
            vals = []
            for r in rr:
                Sx = conclusions(D, [c, c + "#r"], {c: V[c], c + "#r": r}, meas_by)
                f, _ = flip_units(D, [c, c + "#r"], Sx, ids, [c, c + "#r"], margins)
                vals.append(f)
            f = np.concatenate(vals)
            bs = [rng.choice(f, len(f)).mean() for _ in range(B)] if len(f) else [float("nan")]
            nf[c] = {"flip": float(f.mean()) if len(f) else float("nan"), "upper_95": float(np.percentile(bs, 97.5))}
        main["noise"] = nf
        main["noise_upper_95"] = max(v["upper_95"] for v in nf.values())
    main["H1"] = len(R) >= 2
    main["H2"] = bool(main["perm_p"] < 0.05 and (not reps or main["flip"] > main["noise_upper_95"]))
    main["H3"] = bool(main["slope_perm_p"] < 0.05)
    main["eligible"] = cfgs
    return main
