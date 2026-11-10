"""Run judge configs over pilot items via NVIDIA NIM, OpenRouter (free) and the local Claude CLI. Every call is cached as JSON in cache/.

Keys: NVIDIA_API_KEY / OPENROUTER_API_KEY from the environment, or from .env (or the file named by JR_ENV_FILE).
Only free-tier endpoints are used. OpenRouter calls pin one provider (allow_fallbacks=false) and log who served them.
Usage: uv run python src/run_judges.py [--configs c1,c2] [--rep 0]
"""
import argparse, datetime, hashlib, json, os, re, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(os.environ.get("JR_ENV_FILE", ROOT / ".env"))

# key -> (backend, model id, pinned OpenRouter provider or None). One model per family.
JUDGE_MODELS = {
    "deepseek": ("nim", "deepseek-ai/deepseek-v4.1-flash", None),
    "mistral": ("nim", "mistralai/mistral-large-4-1t-a52b", None),
    "gemma": ("nim", "google/gemma-4-31b-it", None),
    "gptoss": ("nim", "openai/gpt-oss-20b", None),
    "nemotron": ("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", "nvidia"),
    # "Claude via CLI": subscription, pilot only, pairwise only, no temperature control.
    "haiku": ("cli", "claude-haiku-5-5", None),
    "sonnet": ("cli", "claude-sonnet-5-5", None),
}
FORMATS = ["pair", "point"]
CONFIGS = [f"{m}-{f}" for m in JUDGE_MODELS if JUDGE_MODELS[m][0] != "cli" for f in FORMATS] + ["haiku-pair", "sonnet-pair"]
CLI_EMPTY_DIR = tempfile.mkdtemp(prefix="jr-cli-")


CLI_VERSION = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()


class UsageLimit(Exception):
    pass


def cli_call(model, sys_p, user_p):
    cmd = ["claude", "-p", user_p, "--model", model, "--system-prompt", sys_p, "--tools", "", "--strict-mcp-config",
           "--setting-sources", "", "--no-session-persistence", "--output-format", "json"]
    r = subprocess.run(cmd, cwd=CLI_EMPTY_DIR, capture_output=True, text=True, timeout=300)
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"cli exit {r.returncode}: {(r.stderr or r.stdout)[:200]}")
    if d.get("is_error"):
        msg = str(d.get("result", ""))
        if re.search(r"limit|quota|usage", msg, re.I):
            raise UsageLimit(msg[:200])
        raise RuntimeError(msg[:200])
    time.sleep(1)
    return d.get("result") or "", list((d.get("modelUsage") or {}).keys())

PAIR_SYS = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
    "to the user question displayed below. Choose the assistant that follows the user's instructions and answers "
    "the user's question better. Consider helpfulness, relevance, accuracy, depth, creativity, and level of detail. "
    "Avoid position bias, do not let response length influence you, and do not favor assistant names. "
    "Be as objective as possible. After a short explanation, output your final verdict strictly in this format: "
    '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.'
)
PAIR_USER = (
    "[User Question]\n{q}\n\n[The Start of Assistant A's Answer]\n{a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{b}\n[The End of Assistant B's Answer]"
)
POINT_SYS = (
    "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the "
    "user question displayed below. Consider helpfulness, relevance, accuracy, depth, creativity, and level of detail. "
    "Be as objective as possible. After a short explanation, rate the response on a scale of 1 to 10 strictly in "
    'this format: "[[rating]]", for example "Rating: [[5]]".'
)
POINT_USER = "[Question]\n{q}\n\n[The Start of Assistant's Answer]\n{a}\n[The End of Assistant's Answer]"


def client(backend):
    if backend == "cli":
        return "cli"
    if backend == "nim":
        key, url = os.environ.get("NVIDIA_API_KEY"), os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    else:
        key, url = os.environ.get("OPENROUTER_API_KEY"), os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not key:
        sys.exit(f"API key for {backend} not set")
    return OpenAI(base_url=url, api_key=key, max_retries=0, timeout=180)


def call(cli, model, provider, sys_p, user_p, cache_path, nonce=""):
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    msgs = [{"role": "system", "content": sys_p + nonce}, {"role": "user", "content": user_p}]
    for attempt in range(6):
        try:
            if cli == "cli":
                text, served = cli_call(model, sys_p + nonce, user_p)
                if not re.search(r"\[\[", text) and attempt < 2:
                    raise RuntimeError("empty or verdict-less reply")
                out = {"model": model, "returned_model": served, "provider": "claude-cli", "cli_version": CLI_VERSION,
                       "date": datetime.datetime.utcnow().isoformat(), "text": text, "nonce": nonce}
                break
            extra = {"provider": {"order": [provider], "allow_fallbacks": False}} if provider else None
            r = cli.chat.completions.create(model=model, messages=msgs, temperature=0, max_tokens=4096, extra_body=extra)
            text = r.choices[0].message.content or ""
            if not re.search(r"\[\[", text) and attempt < 2:
                raise RuntimeError("empty or verdict-less reply")
            out = {"model": model, "returned_model": r.model, "provider": (r.model_extra or {}).get("provider", "nim"),
                   "date": datetime.datetime.utcnow().isoformat(), "text": text, "nonce": nonce,
                   "finish": r.choices[0].finish_reason}
            break
        except UsageLimit:
            raise
        except Exception as e:  # rate limits and transient 5xx
            if attempt == 5:
                out = {"model": model, "date": datetime.datetime.utcnow().isoformat(), "text": "", "error": str(e)[:300]}
            else:
                time.sleep(min(2 ** attempt * 5, 120))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out))
    return out


def parse_pair(t):
    m = re.findall(r"\[\[([ABC])\]\]", t)
    return {"A": "model_a", "B": "model_b", "C": "tie"}[m[-1]] if m else None


def parse_point(t):
    m = re.findall(r"\[\[(\d+(?:\.\d+)?)\]\]", t)
    return float(m[-1]) if m else None


def tasks_for(cfg, items, rep):
    mkey, fmt = cfg.split("-")
    _, model, provider = JUDGE_MODELS[mkey]
    nonce = "" if rep == 0 else f"\n(run {rep})"
    for it in items:
        if fmt == "pair":
            a, b = (it["ans_b"], it["ans_a"]) if it["swap"] else (it["ans_a"], it["ans_b"])
            p = ROOT / "cache" / f"rep{rep}" / cfg / f"{it['id']}.json"
            yield model, provider, PAIR_SYS, PAIR_USER.format(q=it["question"], a=a, b=b), p, nonce
        else:
            for side in ("a", "b"):
                ans = it[f"ans_{side}"]
                h = hashlib.sha1((it["question"] + ans).encode()).hexdigest()[:12]
                p = ROOT / "cache" / f"rep{rep}" / f"{mkey}-point" / f"{it['qid']}_{h}.json"
                yield model, provider, POINT_SYS, POINT_USER.format(q=it["question"], a=ans), p, nonce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    items = json.loads((ROOT / "data" / "pilot_items.json").read_text())[: args.limit]
    for cfg in args.configs.split(","):
        cli = client(JUDGE_MODELS[cfg.split("-")[0]][0])
        ts = list(tasks_for(cfg, items, args.rep))
        if cli == "cli":  # sequential; stop this config on a usage limit
            res = []
            try:
                for t in ts:
                    res.append(call(cli, *t))
            except UsageLimit as e:
                print(f"{cfg}: usage limit after {len(res)}/{len(ts)} calls: {e}", flush=True)
                continue
        else:
            with ThreadPoolExecutor(args.workers) as ex:
                res = list(ex.map(lambda t: call(cli, *t), ts))
        errs = sum(1 for r in res if r.get("error"))
        print(f"{cfg} rep{args.rep}: {len(res)} calls, {errs} errors", flush=True)


if __name__ == "__main__":
    main()
