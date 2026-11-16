"""Arena analysis in one command: eligibility table, primary path with the noise floor, every sensitivity path, figures.
Reads only the verdict cache. Configs whose rep 0 is incomplete are reported and left out (deviation 2026-11-13h).
Usage: uv run python src/arena_release2.py [--configs gpt-pair,...] [--out results/arena_release2] [--boot 2000] [--perm 1000]
"""
import argparse, json

import numpy as np

import arena_run as ar
import arena_sensitivity as sv
import run_judges as rj
from arena_analyze import analyze
from judge_rashomon.figures import all_figures

ar.JUDGES.setdefault("gpt", "gpt-4o-mini-2024-07-18")  # gpt verdicts come from arena_openai_batch, same cache layout
ALL = [f"{j}-{f}" for j in ("gpt", "nemotron", "mistral", "deepseek") for f in ("pair", "point")]


def rev_verdicts(cfg, items):
    """Canonical verdicts for the both-orders subset judged with the presentation coin flipped."""
    out = np.full(len(items), -1)
    for i, it in enumerate(items):
        p = ar.CACHE / "rep0" / f"{cfg}-rev" / f"{it['id']}.json"
        if it["both_orders"] and p.exists():
            v = rj.parse_pair(json.loads(p.read_text()).get("text", ""))
            out[i] = -2 if v is None else 2 if v == "tie" else (0 if v == "model_a" else 1) ^ it["swap"] ^ 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(ALL))
    ap.add_argument("--out", default="results/arena_release2")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--perm", type=int, default=1000)
    a = ap.parse_args()
    items = ar.load_items()
    meta = json.loads((ar.ROOT / "data" / "arena_pairs.json").read_text())["pairs"]
    meta10 = json.loads((ar.ROOT / "data" / "arena_pairs_10w.json").read_text())["pairs"]
    meas = np.array([it["half"] == "measure" for it in items])
    V, table = {}, {}
    for c in a.configs.split(","):
        v = ar.verdicts(c, items, 0)
        ran = v != -1
        table[c] = {"coverage": float(ran.mean()), "failures": float((v[ran] == -2).mean()) if ran.any() else float("nan"),
                    "complete": bool(ran.all())}
        print(f"{c:15} coverage {table[c]['coverage']:.3f}  failures {table[c]['failures']:.3f}  {'' if ran.all() else 'INCOMPLETE, left out'}")
        if ran.all():
            V[c] = v
    reps = {}
    for c in V:  # noise floor: re-runs exist only for the configs phase_noise picked
        rr = [ar.verdicts(c, items, r) for r in (1, 2)]
        if all((r[meas] != -1).all() for r in rr):
            reps[c] = rr
    print("noise-floor configs:", list(reps) or "none yet")
    Vrev = {c: rev_verdicts(c, items) for c in V if c.endswith("-pair")}
    m = analyze(items, meta, V, reps or None, a.boot, a.perm)
    sens = sv.run(items, meta, V, reps or None, Vrev, meta10, a.boot, a.perm)
    out = ar.ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    enc = lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)
    json.dump({"eligibility": table, "configs": list(V), "noise_configs": list(reps), "primary": m, "sensitivity": sens},
              open(out / "report.json", "w"), indent=1, default=enc)
    json.dump(m, open(out / "primary.json", "w"), indent=1, default=enc)
    all_figures(json.loads(json.dumps(m, default=enc)), out)
    sv.multiverse_figure(sens, out / "multiverse.png")
    print(json.dumps({k: m.get(k) for k in ("eligible", "best", "R", "flip", "perm_p", "noise_upper_95", "flip_all",
                                            "slope_perm_p", "H1", "H2", "H3")}, default=enc))
    print("wrote", out)


if __name__ == "__main__":
    main()
