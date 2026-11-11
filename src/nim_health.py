"""Exit 0 if NIM's Nemotron-3-Ultra endpoint answers NVIDIA's own documented sample request (>= 5 of 6 non-empty), else 1."""
import sys
from concurrent.futures import ThreadPoolExecutor

import run_judges as rj

c = rj.client("nim")


def ok(_):
    try:
        r = c.chat.completions.create(model="nvidia/nemotron-3-ultra-550b-a55b", max_tokens=16000, temperature=1.0, top_p=0.95,
                                      messages=[{"role": "user", "content": "What is the capital of Japan?"}],
                                      extra_body={"chat_template_kwargs": {"enable_thinking": True}})
        return bool((r.choices[0].message.content or "").strip())
    except Exception:
        return False


with ThreadPoolExecutor(3) as ex:
    n = sum(ex.map(ok, range(6)))
print(f"nemotron health {n}/6", flush=True)
sys.exit(0 if n >= 5 else 1)
