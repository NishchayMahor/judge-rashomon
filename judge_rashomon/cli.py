"""judge-rashomon CLI: read judge verdicts + human labels from one CSV, report set, flips, nulls.

CSV columns (one row per item x config):
  item_id, system_1, system_2, human, config, verdict [, split]
  human   = 1 | 2           (which system the human preferred; human ties are dropped)
  verdict = 1 | 2 | tie | fail
  split   = select | measure (optional; default = seeded 50/50 within each system pair)
A config named "<config>#<anything>" is a re-run of <config> and feeds the noise floor.
Usage: judge-rashomon verdicts.csv [--families families.json] [--boot 2000] [--perm 1000] [--out report.json]
"""
import argparse, csv, hashlib, json, sys
from collections import defaultdict

import numpy as np

from . import core

CODE = {"1": 0, "2": 1, "tie": 2, "fail": -2}


def load(path):
    rows = list(csv.DictReader(open(path, newline="")))
    need = {"item_id", "system_1", "system_2", "human", "config", "verdict"}
    if missing := need - set(rows[0] if rows else ()):
        sys.exit(f"missing columns: {sorted(missing)}")
    items, idx = [], {}
    for r in rows:  # one item per item_id, canonical orientation = sorted system names
        if r["item_id"] in idx or r["human"] not in ("1", "2"):
            continue
        a, b = r["system_1"], r["system_2"]
        flip = a > b
        hum = ("second" if r["human"] == "1" else "first") if flip else ("first" if r["human"] == "1" else "second")
        half = r.get("split") or ("select" if int(hashlib.sha1(r["item_id"].encode()).hexdigest(), 16) % 2 else "measure")
        idx[r["item_id"]] = (len(items), flip)
        items.append({"pair": sorted((a, b)), "human": hum, "half": half})
    V, reps = defaultdict(lambda: np.full(len(items), -1)), defaultdict(dict)
    for r in rows:
        if r["item_id"] not in idx:
            continue
        i, flip = idx[r["item_id"]]
        v = CODE.get(r["verdict"].strip().lower(), -2)
        if flip and v in (0, 1):
            v = 1 - v
        base, _, tag = r["config"].partition("#")
        (reps[base].setdefault(tag, np.full(len(items), -1)) if tag else V[base])[i] = v
    by_pair = defaultdict(list)
    for it in items:
        by_pair["|".join(it["pair"])].append(it["human"] == "first")
    meta = {k: {"n_nontie": len(v), "p_first": float(np.mean(v))} for k, v in by_pair.items()}
    return items, meta, dict(V), {c: list(r.values()) for c, r in reps.items() if c in V}


def summary(m):
    f = lambda x: "n/a" if x != x else f"{x:.3f}"
    lines = [
        f"Eligible configs: {', '.join(m['eligible'])}",
        f"Best config: {m['best']}   Rashomon set (within 1 SE of best kappa): {', '.join(m['R'])}",
        f"Flip rate inside set: {f(m['flip'])}  95% CI {[round(x, 3) for x in m['ci']['flip']]}  ({m['n_units']} units)",
        f"Permutation null mean: {f(m['perm_null_mean'])}   p = {f(m['perm_p'])}",
        f"Flip rate across all configs: {f(m['flip_all'])}",
        f"Flip probability at human margin 0.05 / 0.20: {f(m['p_at_005'])} / {f(m['p_at_020'])};"
        f" slope vs null p = {f(m['slope_perm_p'])}",
    ]
    if "noise_upper_95" in m:
        lines.append(f"Re-run noise floor (upper 95%): {f(m['noise_upper_95'])}")
    if len(m["R"]) < 2:
        return "\n".join(lines + ["=> Only one config is within 1 SE of the best: no equally-good set to compare."])
    verdict = ("Validated judges disagree on system conclusions beyond chance" + (" and re-run noise" if "noise" in m else "")
               if m["H2"] else "No evidence that validated judges disagree beyond chance" + (" or re-run noise" if "noise" in m else ""))
    return "\n".join(lines + [f"=> {verdict}."])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="judge-rashomon", description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--families", help="JSON {config: [contestant name prefixes it must not judge]}")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    items, meta, V, reps = load(a.csv)
    fam = json.load(open(a.families)) if a.families else None
    m = core.analyze(items, meta, V, reps or None, B=a.boot, P=a.perm, seed=a.seed, families=fam)
    print(summary(m))
    if a.out:
        json.dump(m, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
