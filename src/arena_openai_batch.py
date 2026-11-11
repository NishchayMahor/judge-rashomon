"""gpt-4o-mini pairwise judge on the Arena items via the OpenAI Batch API (PREREG-arena deviation 2026-11-11b).

Same prompts and parsing as every other pairwise config; temperature 0, max_tokens 1024 (bounds worst-case cost).
Results land in cache_arena/rep{rep}/gpt-pair[-rev]/{id}.json in the same format run_judges.call() writes.
Hard budget: the script refuses to submit if spent + this batch's worst case would exceed BUDGET_USD.
Usage: uv run python src/arena_openai_batch.py submit [--rep 0] [--items measure] | collect <batch_id>
"""
import argparse, datetime, json, os, sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import run_judges as rj
from arena_run import CACHE, load_items

ROOT = rj.ROOT
load_dotenv(ROOT / ".env")
MODEL, MAX_TOK = "gpt-4o-mini-2024-07-18", 1024
PRICE_IN, PRICE_OUT = 0.075e-6, 0.30e-6  # batch tier, USD per token (checked 2026-11-11)
BUDGET_USD = 2.0
SPEND = ROOT / "logs" / "openai_spend.json"


def spent():
    return json.loads(SPEND.read_text())["usd"] if SPEND.exists() else 0.0


def requests(items, rep):
    nonce = "" if rep == 0 else f"\n(run {rep})"
    for it in items:
        for rev in ((False, True) if it["both_orders"] and rep == 0 else (False,)):
            swap = it["swap"] ^ rev
            a, b = (it["ans_second"], it["ans_first"]) if swap else (it["ans_first"], it["ans_second"])
            path = CACHE / f"rep{rep}" / ("gpt-pair" + ("-rev" if rev else "")) / f"{it['id']}.json"
            if path.exists():
                continue
            yield str(path.relative_to(CACHE)), {
                "model": MODEL, "temperature": 0, "max_tokens": MAX_TOK,
                "messages": [{"role": "system", "content": rj.PAIR_SYS + nonce},
                             {"role": "user", "content": rj.PAIR_USER.format(q=it["prompt"], a=a, b=b)}]}


def submit(rep, half):
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    items = [it for it in load_items() if half == "all" or it["half"] == half]
    reqs = list(requests(items, rep))
    tin = sum(len(enc.encode(m["content"])) + 4 for _, b in reqs for m in b["messages"])
    worst = tin * PRICE_IN + len(reqs) * MAX_TOK * PRICE_OUT
    print(f"{len(reqs)} requests, {tin} input tokens, worst case ${worst:.2f}, spent so far ${spent():.2f}")
    if spent() + worst > BUDGET_USD:
        sys.exit(f"refusing: spent ${spent():.2f} + worst case ${worst:.2f} > budget ${BUDGET_USD}")
    f = ROOT / "logs" / f"openai_batch_rep{rep}.jsonl"
    f.write_text("".join(json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": b}) + "\n"
                         for cid, b in reqs))
    c = OpenAI()
    up = c.files.create(file=f.open("rb"), purpose="batch")
    bt = c.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h")
    print(f"BATCH {bt.id}")


def collect(batch_id):
    c = OpenAI()
    bt = c.batches.retrieve(batch_id)
    print(f"status {bt.status} counts {bt.request_counts}")
    if bt.status != "completed":
        return
    usd = 0.0
    for fid, ok in ((bt.output_file_id, True), (bt.error_file_id, False)):
        if not fid:
            continue
        for line in c.files.content(fid).text.splitlines():
            r = json.loads(line)
            p = CACHE / r["custom_id"]
            p.parent.mkdir(parents=True, exist_ok=True)
            body = (r.get("response") or {}).get("body") or {}
            now = datetime.datetime.utcnow().isoformat()
            if ok and r.get("response", {}).get("status_code") == 200:
                u = body["usage"]
                usd += u["prompt_tokens"] * PRICE_IN + u["completion_tokens"] * PRICE_OUT
                out = {"model": MODEL, "returned_model": body["model"], "provider": "openai-batch", "batch": batch_id,
                       "date": now, "text": body["choices"][0]["message"]["content"] or "", "nonce": "",
                       "finish": body["choices"][0]["finish_reason"]}
            else:
                out = {"model": MODEL, "date": now, "text": "", "batch": batch_id,
                       "error": str(r.get("error") or body.get("error") or r.get("response", {}).get("status_code"))[:300]}
            p.write_text(json.dumps(out))
    SPEND.write_text(json.dumps({"usd": spent() + usd}))
    print(f"collected; this batch ${usd:.3f}, total ${spent():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "collect"])
    ap.add_argument("batch_id", nargs="?")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--items", default="all", choices=["all", "select", "measure"])
    a = ap.parse_args()
    submit(a.rep, a.items) if a.cmd == "submit" else collect(a.batch_id)
