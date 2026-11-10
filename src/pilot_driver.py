"""Unattended pilot finish: wait for rep0, apply the gpt-oss cutoff, run the noise floor, then analyze.

Noise configs = best pairwise and best pointwise by kappa among non-CLI configs (keeps the Claude CLI pilot-sized).
Usage: uv run python src/pilot_driver.py --cutoff 2026-11-11T09:00
"""
import argparse, datetime, json, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze import kappa, load_verdicts
from run_judges import CONFIGS, JUDGE_MODELS

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "driver.log"


def log(msg):
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def running(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True)
    args = ap.parse_args()
    cutoff = datetime.datetime.fromisoformat(args.cutoff)
    while running("src/run_judges.py"):
        if datetime.datetime.now() >= cutoff and running("configs gptoss"):
            subprocess.run(["pkill", "-f", "configs gptoss"])
            log("cutoff reached: stopped gpt-oss (amendment)")
        time.sleep(60)
    log("rep0 done")
    items = json.loads((ROOT / "data" / "pilot_items.json").read_text())
    idx = range(len(items))
    ks = {c: kappa(items, load_verdicts(ROOT / "cache", c, 0, items), idx) for c in CONFIGS
          if JUDGE_MODELS[c.split("-")[0]][0] != "cli" and (ROOT / "cache" / "rep0" / c).exists()}
    noise = [max((c for c in ks if c.endswith(f)), key=ks.get) for f in ("-pair", "-point")]
    log(f"noise configs {noise} kappas {[round(ks[c], 3) for c in noise]}")
    for rep in (1, 2):
        procs = [subprocess.Popen(["uv", "run", "python", "src/run_judges.py", "--configs", c, "--rep", str(rep),
                                   "--workers", "6"], cwd=ROOT) for c in noise]
        for p in procs:
            p.wait()
        log(f"rep{rep} done")
    for extra, name in (([], "analyze.out"), (["--no-cli", "--out", "pilot_metrics_nocli.json"], "analyze_nocli.out")):
        subprocess.run(["uv", "run", "python", "src/analyze.py", "--noise", ",".join(noise), *extra], cwd=ROOT,
                       stdout=(ROOT / "logs" / name).open("w"), stderr=subprocess.STDOUT)
    log("analysis done")


if __name__ == "__main__":
    main()
