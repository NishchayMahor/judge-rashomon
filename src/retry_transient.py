"""One retry pass over transient failures in rep0 (PREREG deviation D-2026-11-10b).

Transient = timeout, connection error, 429, 5xx. The failed entry is moved to cache_failed/ for audit,
then the same task is re-sent once through the standard call() wrapper with identical prompt, params and provider pin.
Usage: uv run python src/retry_transient.py --configs deepseek-pair,...
"""
import argparse, json, re, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from run_judges import CONFIGS, JUDGE_MODELS, ROOT, call, client, tasks_for

TRANSIENT = re.compile(r"timed out|Connection error|Error code: (429|5\d\d)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(c for c in CONFIGS if JUDGE_MODELS[c.split("-")[0]][0] != "cli"))
    args = ap.parse_args()
    items = json.loads((ROOT / "data" / "pilot_items.json").read_text())
    for cfg in args.configs.split(","):
        cli = client(JUDGE_MODELS[cfg.split("-")[0]][0])
        todo = []
        for t in tasks_for(cfg, items, 0):
            p = t[4]
            if p.exists() and TRANSIENT.search(str(json.loads(p.read_text()).get("error", ""))):
                dst = ROOT / "cache_failed" / p.relative_to(ROOT / "cache")
                if dst.exists():  # already retried once; never twice
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(p, dst)
                todo.append(t)
        with ThreadPoolExecutor(6) as ex:
            res = list(ex.map(lambda t: call(cli, *t), todo))
        print(f"{cfg}: retried {len(todo)}, still failing {sum(1 for r in res if r.get('error'))}", flush=True)


if __name__ == "__main__":
    main()
