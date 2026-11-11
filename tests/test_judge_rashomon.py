"""Known-answer tests: the null is calibrated, a planted disagreement is found, CSV orientation is handled."""
import csv

import numpy as np

from judge_rashomon import cli, core, synth


def run(tmp_path, planted, seed):
    p = tmp_path / f"v{planted}{seed}.csv"
    synth.make(planted, seed=seed, path=p)
    items, meta, V, reps = cli.load(p)
    return core.analyze(items, meta, V, None, B=50, P=100, seed=seed)


def test_null_rarely_rejects(tmp_path):
    ps = [run(tmp_path, False, s)["perm_p"] for s in range(10)]
    assert np.mean([p < 0.05 for p in ps if p == p]) <= 0.2


def test_planted_disagreement_is_found(tmp_path):
    ms = [run(tmp_path, True, s) for s in range(5)]
    found = [m["perm_p"] < 0.05 for m in ms if len(m["R"]) >= 2]
    assert found and np.mean(found) >= 0.6


def test_swapped_columns_give_same_answer(tmp_path):
    rows = synth.make(True, n_pairs=10, seed=3)
    swapped = [dict(r, system_1=r["system_2"], system_2=r["system_1"],
                    human={"1": "2", "2": "1"}[r["human"]], verdict={"1": "2", "2": "1"}.get(r["verdict"], r["verdict"]))
               for r in rows]
    out = []
    for name, rs in (("a", rows), ("b", swapped)):
        p = tmp_path / f"{name}.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
        items, meta, V, _ = cli.load(p)
        out.append(core.analyze(items, meta, V, None, B=10, P=10, seed=0))
    assert out[0]["R"] == out[1]["R"] and out[0]["flip"] == out[1]["flip"]


def test_family_rule_drops_units(tmp_path):
    p = tmp_path / "f.csv"
    synth.make(False, n_pairs=10, seed=1, path=p)
    items, meta, V, _ = cli.load(p)
    base = core.analyze(items, meta, V, None, B=5, P=5, seed=0)
    fam = {c: ("sys000", "sys001") for c in V}
    fewer = core.analyze(items, meta, V, None, B=5, P=5, seed=0, families=fam)
    assert fewer["n_units"] < base["n_units"]
