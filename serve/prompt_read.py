"""Cold prompt reads, one at a time: N distinct random-word prompts of WORDS words (nothing to reuse), one token
each; prints the prompt's tokens, the wait for the first token and the rate that gives.

    prompt_read.py N WORDS [BASE_URL] [SEED]"""
import json
import random
import sys
import time
import urllib.request

n, n_words = int(sys.argv[1]), int(sys.argv[2])
base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8095/v1"
seed = int(sys.argv[4]) if len(sys.argv) > 4 else int(time.time())
words = "server cache kernel tensor socket memory request decode prefill expert layer token batch lane queue".split()

for i in range(n):
    rnd = random.Random(seed * 1000 + i)
    text = f"Note {seed}-{i}.\n" + " ".join(rnd.choice(words) for _ in range(n_words)) + "\nSay OK."
    body = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": 1, "stream": True,
            "stream_options": {"include_usage": True}, "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, first, usage = time.time(), None, None
    for line in urllib.request.urlopen(req, timeout=3600):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            usage = d["usage"]
        if first is None and d.get("choices") and d["choices"][0]["delta"].get("content"):
            first = time.time()
    if usage is None or first is None:
        raise SystemExit(f"request {i}: no usage or no content in the stream")
    p = usage["prompt_tokens"]
    print(f"read {i}: {p} tokens, first token after {first - t0:6.2f} s -> {p / (first - t0):7.1f} tok/s")
