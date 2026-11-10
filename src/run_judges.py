"""Run judge configs over pilot items via NVIDIA NIM. Every call is cached as JSON in cache/.

Key: NVIDIA_API_KEY from the environment, or from the dotenv file named by JR_ENV_FILE.
Usage: uv run python src/run_judges.py [--configs c1,c2] [--rep 0]
"""
import argparse, datetime, hashlib, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
if os.environ.get("JR_ENV_FILE"):
    load_dotenv(os.environ["JR_ENV_FILE"])

JUDGE_MODELS = {
    "llama70": "meta/llama-3.3-70b-instruct",
    "llama8": "meta/llama-3.1-8b-instruct",
    "qwen": "qwen/qwen2.5-coder-32b-instruct",
    "mistral": "mistralai/mixtral-8x22b-instruct-v0.1",
    "gemma": "google/gemma-2-27b-it",
}
FORMATS = ["pair", "point"]
CONFIGS = [f"{m}-{f}" for m in JUDGE_MODELS for f in FORMATS]

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


def client():
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        sys.exit("NVIDIA_API_KEY not set (export it or point JR_ENV_FILE at a dotenv file)")
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)


def call(cli, model, sys_p, user_p, cache_path, nonce=""):
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    msgs = [{"role": "system", "content": sys_p + nonce}, {"role": "user", "content": user_p}]
    for attempt in range(6):
        try:
            r = cli.chat.completions.create(model=model, messages=msgs, temperature=0, max_tokens=1024)
            out = {"model": model, "returned_model": r.model, "date": datetime.datetime.utcnow().isoformat(),
                   "text": r.choices[0].message.content or "", "nonce": nonce}
            break
        except Exception as e:  # rate limits and transient 5xx
            if attempt == 5:
                out = {"model": model, "date": datetime.datetime.utcnow().isoformat(), "text": "", "error": str(e)[:300]}
            else:
                time.sleep(2 ** attempt * 3)
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
    model = JUDGE_MODELS[mkey]
    nonce = "" if rep == 0 else f"\n(run {rep})"
    for it in items:
        if fmt == "pair":
            a, b = (it["ans_b"], it["ans_a"]) if it["swap"] else (it["ans_a"], it["ans_b"])
            p = ROOT / "cache" / f"rep{rep}" / cfg / f"{it['id']}.json"
            yield model, PAIR_SYS, PAIR_USER.format(q=it["question"], a=a, b=b), p, nonce
        else:
            for side in ("a", "b"):
                ans = it[f"ans_{side}"]
                h = hashlib.sha1((it["question"] + ans).encode()).hexdigest()[:12]
                p = ROOT / "cache" / f"rep{rep}" / f"{mkey}-point" / f"{it['qid']}_{h}.json"
                yield model, POINT_SYS, POINT_USER.format(q=it["question"], a=ans), p, nonce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    items = json.loads((ROOT / "data" / "pilot_items.json").read_text())[: args.limit]
    cli = client()
    for cfg in args.configs.split(","):
        ts = list(tasks_for(cfg, items, args.rep))
        with ThreadPoolExecutor(args.workers) as ex:
            res = list(ex.map(lambda t: call(cli, *t), ts))
        errs = sum(1 for r in res if r.get("error"))
        print(f"{cfg} rep{args.rep}: {len(res)} calls, {errs} errors", flush=True)


if __name__ == "__main__":
    main()
