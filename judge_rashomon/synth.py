"""Synthetic datasets with a known answer, for tests and the demo.

null:    configs are exchangeable (each follows the human vote w.p. acc, independently).
planted: on close system pairs (|p - 0.5| < 0.1) configs split into two camps preferring opposite systems; agreement
         with humans is equal across camps by construction.
"""
import csv

import numpy as np


def make(planted, n_pairs=40, per_pair=40, n_configs=6, acc=0.75, seed=0, path=None):
    rng = np.random.default_rng(seed)
    p_first = rng.uniform(0.3, 0.7, n_pairs)
    rows = []
    for k in range(n_pairs):
        a, b = f"sys{2 * k:03d}", f"sys{2 * k + 1:03d}"
        for j in range(per_pair):
            item = f"p{k}-i{j}"
            h = 0 if rng.random() < p_first[k] else 1
            split = "select" if j < per_pair // 2 else "measure"
            for c in range(n_configs):
                v = h if rng.random() < acc else 1 - h
                if planted and abs(p_first[k] - 0.5) < 0.1:
                    fav = (k + c) % 2
                    v = fav if rng.random() < acc else 1 - fav
                rows.append({"item_id": item, "system_1": a, "system_2": b, "human": str(h + 1), "config": f"cfg{c}",
                             "verdict": str(v + 1), "split": split})
    if path:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    return rows
