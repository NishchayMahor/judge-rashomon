"""Pre-freeze dry run: does the primary path recover a known flip structure? (PREREG-arena checklist item.)

Uses the real Arena items and human labels, with synthetic judge verdicts for 6 configs (Gemma excluded, see PREREG-arena).
- null: configs are exchangeable (each follows the human vote w.p. 0.75, independently). H2 should fail at ~5%.
- planted: on close pairs (margin < 0.1) configs split into two camps that prefer opposite models; agreement with
  humans is equal across camps by construction. H2 should pass and the flip-vs-margin slope should be negative.
Usage: uv run python src/arena_synth.py [--runs 20] [--boot 200] [--perm 200]
"""
import argparse, json
from pathlib import Path

import numpy as np

from arena_analyze import Data, analyze

ROOT = Path(__file__).resolve().parent.parent
CFGS = [f"{j}-{f}" for j in ("deepseek", "mistral", "nemotron") for f in ("pair", "point")]


def synth(D, rng, planted):
    margins = np.abs(D.p_first - 0.5)[D.pair_of]
    V, reps = {}, {}
    for k, c in enumerate(CFGS):
        def draw():
            v = np.where(rng.random(len(D.h)) < 0.75, D.h, 1 - D.h)
            if planted:
                close = margins < 0.1
                camp = k % 2  # pair/point alternate camps, so each camp spans all judges
                fav = np.where((D.pair_of % 2) == camp, 0, 1)  # camp-specific favourite per pair
                pick = rng.random(len(D.h)) < 0.75
                v = np.where(close, np.where(pick, fav, 1 - fav), v)
            v = np.where(rng.random(len(D.h)) < 0.05, 2, v)  # some judge ties
            v = np.where(rng.random(len(D.h)) < 0.01, -2, v)  # some failures
            return v
        V[c] = draw()
        reps[c] = [draw(), draw()] if c in ("nemotron-pair", "nemotron-point") else None
    return V, {c: r for c, r in reps.items() if r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--perm", type=int, default=200)
    a = ap.parse_args()
    items = json.loads((ROOT / "data" / "arena_items.json").read_text())
    meta = json.loads((ROOT / "data" / "arena_pairs.json").read_text())["pairs"]
    D = Data(items, meta)
    out = {}
    for planted in (False, True):
        res = []
        for r in range(a.runs):
            V, reps = synth(D, np.random.default_rng(1000 * planted + r), planted)
            m = analyze(items, meta, V, reps, B=a.boot, P=a.perm, seed=r)
            res.append({k: m[k] for k in ("flip", "perm_null_mean", "perm_p", "noise_upper_95", "slope", "perm_null_slope_mean", "slope_perm_p", "H1", "H2", "H3")}
                       | {"R": len(m["R"])})
        name = "planted" if planted else "null"
        out[name] = {"runs": res, "H2_rate": float(np.mean([x["H2"] for x in res])),
                     "H3_rate": float(np.mean([x["H3"] for x in res])), "mean_flip": float(np.mean([x["flip"] for x in res])),
                     "mean_R": float(np.mean([x["R"] for x in res]))}
        print(name, {k: v for k, v in out[name].items() if k != "runs"}, flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "arena_synth_dryrun.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
