"""Unattended pilot finish: wait for rep0, apply the gpt-oss cutoff, run the noise floor, then analyze.

Noise configs = best pairwise and best pointwise by complete-case kappa among eligible non-CLI configs.
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
    r = subprocess.run(["uv", "run", "python", "src/retry_transient.py"], cwd=ROOT, capture_output=True, text=True)
    log("transient retry (deviation 2026-11-10b): " + r.stdout.strip().replace("\n", "; "))
    items = json.loads((ROOT / "data" / "pilot_items.json").read_text())
    ks = {}
    for c in CONFIGS:  # non-CLI, >=90% coverage, <=5% failures, kappa on its own complete cases (amendment A5/A6)
        if JUDGE_MODELS[c.split("-")[0]][0] == "cli" or not (ROOT / "cache" / "rep0" / c).exists():
            continue
        v = load_verdicts(ROOT / "cache", c, 0, items)
        done = [x for x in v if x is not None]
        if len(done) < 0.9 * len(items) or sum(x == "FAIL" for x in done) > 0.05 * len(done):
            continue
        ks[c] = kappa(items, v, [i for i, x in enumerate(v) if x not in (None, "FAIL")])
    noise = [max((c for c in ks if c.endswith(f)), key=ks.get) for f in ("-pair", "-point")]
    log(f"noise configs {noise} kappas {[round(ks[c], 3) for c in noise]}")
    for rep in (1, 2):
        procs = [subprocess.Popen(["uv", "run", "python", "src/run_judges.py", "--configs", c, "--rep", str(rep),
                                   "--workers", "6"], cwd=ROOT) for c in noise]
        for p in procs:
            p.wait()
        log(f"rep{rep} done")
    paths = (([], "pilot_metrics.json"), (["--with-cli"], "pilot_metrics_withcli.json"),
             (["--ties-as-tie"], "pilot_metrics_tiesastie.json"))
    for extra, out in paths:  # primary path first, then pre-declared sensitivity paths
        subprocess.run(["uv", "run", "python", "src/analyze.py", "--noise", ",".join(noise), "--out", out, *extra],
                       cwd=ROOT, stdout=(ROOT / "logs" / f"{out}.log").open("w"), stderr=subprocess.STDOUT)
    log("analysis done")


if __name__ == "__main__":
    main()
