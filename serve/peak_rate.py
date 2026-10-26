"""Decode rate over a streamed reply: the whole run's, and the best sliding windows (1 s and 3 s) of it, for C requests
at once (bench_steady's prompts: random words, thinking off). A streamed chunk can carry several tokens (accepted
drafts), so each request's chunks are weighted by its completion_tokens / chunks, as bench_steady does.

    peak_rate.py C PROMPT_WORDS MAX_TOKENS [BASE_URL]"""
import json
import random
import sys
import threading
import time
import urllib.request

C, n_words, n_gen = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
base = sys.argv[4] if len(sys.argv) > 4 else "http://127.0.0.1:8095/v1"
words = "server cache kernel tensor socket memory request decode prefill expert layer token batch lane queue".split()
res = [None] * C


def one(i):
    rnd = random.Random(7000 * C + i)
    text = " ".join(rnd.choice(words) for _ in range(n_words)) + f"\nRequest {i}: write a long essay about GPUs."
    body = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": n_gen, "stream": True,
            "stream_options": {"include_usage": True}, "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False}, "temperature": 1.0, "top_p": 0.95}
    req = urllib.request.Request(f"{base}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    stamps, usage = [], None
    for line in urllib.request.urlopen(req, timeout=7200):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            usage = d["usage"]
        if d.get("choices") and d["choices"][0]["delta"].get("content"):
            stamps.append(time.time())
    if usage is None or not stamps:
        raise SystemExit(f"request {i}: no usage or no content in the stream")
    res[i] = (stamps, usage["completion_tokens"])


ts = [threading.Thread(target=one, args=(i,)) for i in range(C)]
[t.start() for t in ts]
[t.join() for t in ts]
events = sorted((s, n / len(st)) for st, n in res for s in st)   # (time, tokens) of every chunk of every request
t0, t1 = events[0][0], events[-1][0]
total = sum(w for _, w in events)


def best(win):
    top, j, acc = 0.0, 0, 0.0
    for i, (s, w) in enumerate(events):
        acc += w
        while events[j][0] < s - win:
            acc -= events[j][1]
            j += 1
        if s - t0 >= win:
            top = max(top, acc / win)
    return top


print(f"C={C}: first to last token {total / (t1 - t0):6.1f} tok/s over {t1 - t0:5.1f} s | best 3 s {best(3.0):6.1f} tok/s | "
      f"best 1 s {best(1.0):6.1f} tok/s")
