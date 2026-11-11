"""Build the Arena item set for the full run (PREREG-arena.md, Data section).

Pools the open LMArena 55k/100k/140k releases, applies the filters in order, keeps model pairs with >= 100 non-tie
votes, samples 40 non-tie battles per pair (seed 0), splits them 20/20 into selection/measurement halves, balances the
presentation-order coin within each pair, and marks a seeded 10% both-orders subset.
Usage: uv run python src/arena_items.py
"""
import hashlib, json, random, re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
MIN_VOTES, PER_PAIR, MIN_PROMPT_TOK, MAX_RESP_TOK = 100, 40, 5, 4000


def ntok(s):
    return len(s.split())


def english_55k(s):  # 55k has no language field: ASCII share of letters as a proxy
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) >= 0.95


def text_140k(msg):
    return "".join(p.get("text") or "" for p in msg["content"] if p.get("type") == "text")


def rows():
    """Yield (source, id, model_a, model_b, winner, prompt, resp_a, resp_b) for English single-turn battles."""
    d = load_dataset("lmarena-ai/arena-human-preference-55k", cache_dir=str(ROOT / "scratch_arena"), split="train")
    for r in d:
        p, a, b = (json.loads(r[k]) for k in ("prompt", "response_a", "response_b"))
        if len(p) != 1 or not english_55k(p[0]) or a[0] is None or b[0] is None:
            continue
        w = "model_a" if r["winner_model_a"] else "model_b" if r["winner_model_b"] else "tie"
        yield "55k", str(r["id"]), r["model_a"], r["model_b"], w, p[0], a[0], b[0]
    d = load_dataset("lmarena-ai/arena-human-preference-100k", cache_dir=str(ROOT / "scratch_arena"), split="train")
    for r in d:
        if r["turn"] != 1 or r["language"] != "English" or len(r["conversation_a"]) != 2 or len(r["conversation_b"]) != 2:
            continue
        yield ("100k", r["question_id"], r["model_a"], r["model_b"], r["winner"], r["conversation_a"][0]["content"],
               r["conversation_a"][1]["content"], r["conversation_b"][1]["content"])
    d = load_dataset("lmarena-ai/arena-human-preference-140k", cache_dir=str(ROOT / "scratch_arena"), split="train")
    for r in d:
        ca, cb = r["conversation_a"], r["conversation_b"]
        if r["language"] != "en" or len(ca) != 2 or len(cb) != 2:
            continue
        yield "140k", r["id"], r["model_a"], r["model_b"], r["winner"], text_140k(ca[0]), text_140k(ca[1]), text_140k(cb[1])


def main():
    counts = defaultdict(int)
    seen, battles = set(), defaultdict(list)
    for src, bid, ma, mb, w, p, ra, rb in rows():
        counts["english_single_turn"] += 1
        if not p or not ra or not rb or ntok(p) < MIN_PROMPT_TOK:
            continue
        counts["prompt_ge_5_tok"] += 1
        if max(ntok(ra), ntok(rb)) > MAX_RESP_TOK:
            continue
        counts["resp_le_4000_tok"] += 1
        pair = tuple(sorted((ma, mb)))
        key = (pair, re.sub(r"\s+", " ", p.strip()))
        if key in seen:
            continue
        seen.add(key)
        counts["dedup_prompt_within_pair"] += 1
        w = {"tie (bothbad)": "tie", "both_bad": "tie"}.get(w, w)
        if w != "tie":  # orient the winner to the sorted pair
            w = "first" if (w == "model_a") == (ma == pair[0]) else "second"
        a, b = (ra, rb) if ma == pair[0] else (rb, ra)
        battles[pair].append({"source": src, "battle_id": bid, "prompt": p, "ans_first": a, "ans_second": b, "human": w})
    # no per-user field exists in any release, so the vote cap is not applied
    pairs = {}
    for pair, bs in battles.items():
        nt = [b for b in bs if b["human"] != "tie"]
        if len(nt) >= MIN_VOTES:
            wins = sum(b["human"] == "first" for b in nt)
            pairs[pair] = {"n_nontie": len(nt), "n_tie": len(bs) - len(nt), "p_first": wins / len(nt),
                           "human_first_wins": wins}
    rng = random.Random(0)
    items = []
    for pair in sorted(pairs):
        nt = sorted((b for b in battles[pair] if b["human"] != "tie"), key=lambda b: (b["source"], b["battle_id"]))
        pick = rng.sample(nt, PER_PAIR)
        order = sorted(pick, key=lambda b: hashlib.sha1(f"half|{b['source']}|{b['battle_id']}".encode()).hexdigest())
        swaps = [1] * (PER_PAIR // 2) + [0] * (PER_PAIR // 2)
        rng.shuffle(swaps)
        for i, b in enumerate(order):
            items.append({"id": hashlib.sha1(f"{b['source']}|{b['battle_id']}".encode()).hexdigest()[:16],
                          "pair": list(pair), "half": "select" if i < PER_PAIR // 2 else "measure",
                          "swap": swaps[i], "both_orders": False, **b})
    for it in rng.sample(items, len(items) // 10):
        it["both_orders"] = True
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "arena_items.json").write_text(json.dumps(items))
    meta = {"filter_counts": dict(counts), "n_pairs_any": len(battles), "n_eligible_pairs": len(pairs),
            "n_items": len(items), "pairs": {"|".join(k): v for k, v in sorted(pairs.items())}}
    (ROOT / "data" / "arena_pairs.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "pairs"}, indent=1))


if __name__ == "__main__":
    main()
