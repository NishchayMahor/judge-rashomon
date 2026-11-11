"""Primary path of PREREG-arena.md. The analysis lives in judge_rashomon.core; this module pins the Arena judge
families (self-preference rule) so results are identical to the frozen version (regression-checked 2026-11-11)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from judge_rashomon import core
from judge_rashomon.core import Data, eligible, kappa_from, per_pair_conf  # noqa: F401  (used by arena_run)

JUDGE_FAMILIES = {  # judge -> contestant name prefixes from the same family
    "deepseek": ("deepseek",), "mistral": ("mistral", "mixtral"), "gemma": ("gemma", "gemini"), "nemotron": ("nemotron",),
    "gpt": ("gpt", "chatgpt"),
}


def analyze(items, pairs_meta, V, reps=None, B=2000, P=1000, seed=0):
    fam = {c: JUDGE_FAMILIES[c.split("-")[0]] for c in list(V) + list(reps or {})}
    return core.analyze(items, pairs_meta, V, reps, B, P, seed, families=fam)
