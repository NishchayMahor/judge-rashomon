# judge-rashomon

**Your LLM judge agrees with human labels. Would another judge that agrees with humans just as well rank your systems the same way?**

The usual way to validate a judge is to check its agreement with a few hundred human labels, then trust its rankings. This repo tests whether that's enough. Among judge configs that agree with humans *equally* well (an epsilon-Rashomon set), how often do they reach opposite conclusions about which system is better, beyond chance and re-run noise?

## What's in v0.1
- **The tool** (`judge_rashomon/`): feed it judge verdicts and human labels as a CSV, and it reports the equally-good set, the flip rate, a permutation null, an optional re-run noise floor, and flips against human closeness. See [judge_rashomon/README.md](judge_rashomon/README.md).
- **The MT-Bench pilot**, pre-registered ([PREREG.md](PREREG.md), [RESULTS-pilot.md](RESULTS-pilot.md)): 5 of 6 eligible judge configs were equally good, and they disagreed on 37.6% of items, but **zero** system rankings flipped. MT-Bench's systems are far apart, so the pilot is the easy case.
- **The Arena study, pre-registered and in progress** ([PREREG-arena.md](PREREG-arena.md), with its deviation log): 80 close system pairs from LMArena human votes. Results come in a later release.
- **Paper draft** ([paper/](paper/)): the method and the pilot are written; the Arena sections are placeholders.

## Quickstart
```bash
uv sync
uv run judge-rashomon examples/planted.csv   # planted disagreement: detected
uv run judge-rashomon examples/null.csv      # null data: nothing flagged
uv run pytest
```
Figures from a report: `uv run python -m judge_rashomon.figures report.json figs/`.

## Data
- MT-Bench questions, answers and expert judgments come from Zheng et al. 2023 (`lmsys/mt_bench_human_judgments`, CC-BY-4.0). The pilot items and judge verdict caches in `data/` and `cache/` are derived from them.
- LMArena battles come from the public `lmarena-ai` releases. Only aggregate pair statistics are committed; the item file is rebuilt by `src/arena_items.py`.

## License
Code: MIT. Data derivatives follow their source licenses.
