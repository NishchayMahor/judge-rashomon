"""Release 1 (deviation 2026-11-13h, exploratory): the frozen primary path on configs whose rep 0 is complete.
No noise floor yet (re-runs not done), so H2 here is the permutation test only.
Usage: uv run python src/arena_release1.py [out.json]"""
import json, sys

import numpy as np

import arena_run as ar
from arena_analyze import analyze

ar.JUDGES.setdefault("gpt", "gpt-4o-mini-2024-07-18")  # gpt verdicts come from arena_openai_batch, same cache layout
ALL = ["gpt-pair", "gpt-point", "nemotron-pair", "nemotron-point", "deepseek-pair", "deepseek-point", "mistral-pair", "mistral-point"]

items = ar.load_items()
meta = json.loads((ar.ROOT / "data" / "arena_pairs.json").read_text())["pairs"]
V = {c: ar.verdicts(c, items, 0) for c in ALL}
done = [c for c in ALL if (V[c] != -1).all()]
for c in ALL:
    ran = V[c] != -1
    print(f"{c:15} coverage {ran.mean():.3f}  failures {(V[c][ran] == -2).mean() if ran.any() else float('nan'):.3f}  {'COMPLETE' if c in done else ''}")
m = analyze(items, meta, {c: V[c] for c in done})
m["release1_configs"] = done
out = sys.argv[1] if len(sys.argv) > 1 else "results/arena_release1.json"
ar.ROOT.joinpath(out).parent.mkdir(exist_ok=True)
json.dump(m, open(ar.ROOT / out, "w"), indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x))
print(json.dumps({k: m[k] for k in ("eligible", "best", "R", "flip", "perm_p", "flip_all", "slope_perm_p", "H2") if k in m}, default=float))
