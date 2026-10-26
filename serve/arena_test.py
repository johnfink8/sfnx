"""N long conversations decoding at once, through serve/ against sFNX (the arena: serve/sfnx.json). Each prompt
is WORDS random words (~1.15 tokens a word), so N of them need far more K/V than the pool's own pages hold. First every
prompt is read (one token each, in turn); then all N are asked again at once with a short note after the reply, so
each comes back (from its lane or the conversation cache) and all N decode together at their long contexts. Prints
the second phase as bench_steady.py does: total and steady tok/s (every conversation decoding) and first tokens.

    arena_test.py [BASE_URL] [N WORDS GEN]        (default http://127.0.0.1:8095/v1 6 150000 400)"""
import json
import random
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8095/v1"
N, WORDS, GEN = (int(x) for x in sys.argv[2:5]) if len(sys.argv) > 4 else (6, 150000, 400)
VOCAB = "server cache kernel tensor socket memory request decode prefill expert layer token batch lane queue".split()


def prompt(k):
    rng = random.Random(7000 + k)
    return " ".join(rng.choice(VOCAB) for _ in range(WORDS)) + f"\nConversation {k}: summarise the list above in a few lines."


def ask(messages, max_tokens, out, key):
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}, "temperature": 1.0, "top_p": 0.95}
    req = urllib.request.Request(f"{BASE}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, stamps, text, usage = time.time(), [], [], None
    for line in urllib.request.urlopen(req, timeout=7200):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            usage = d["usage"]
        if d.get("choices") and d["choices"][0]["delta"].get("content"):
            stamps.append(time.time())
            text.append(d["choices"][0]["delta"]["content"])
    if usage is None or not stamps:
        raise SystemExit(f"{key}: no usage or no content in the stream")
    out[key] = (t0, stamps, usage, "".join(text))


def run_all(jobs):
    out, errs = {}, []

    def one(k, messages, max_tokens):
        try:
            ask(messages, max_tokens, out, k)
        except Exception as e:  # noqa: BLE001 - reported below and fails the run
            errs.append(f"{k}: {e!r}")

    ts = [threading.Thread(target=one, args=j) for j in jobs]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise SystemExit("requests failed:\n" + "\n".join(errs))
    return out, time.time() - t0


first, wall1 = run_all([(k, [{"role": "user", "content": prompt(k)}], 1) for k in range(N)])
ptoks = [first[k][2]["prompt_tokens"] for k in range(N)]
print(f"read: {N} prompts of {min(ptoks)}-{max(ptoks)} tokens ({sum(ptoks)} in all) in {wall1:.0f} s", flush=True)
jobs = [(k, [{"role": "user", "content": prompt(k)}, {"role": "assistant", "content": first[k][3]},
             {"role": "user", "content": f"Now write a long essay about GPUs, conversation {k}."}], GEN) for k in range(N)]
res, wall = run_all(jobs)
toks = sum(res[k][2]["completion_tokens"] for k in range(N))
ttft = sorted(res[k][1][0] - res[k][0] for k in range(N))
w0, w1 = max(res[k][1][0] for k in range(N)), min(res[k][1][-1] for k in range(N))
steady = None
if w1 > w0:
    in_win = sum(sum(1 for s in res[k][1] if w0 <= s <= w1) * res[k][2]["completion_tokens"] / len(res[k][1]) for k in range(N))
    steady = in_win / (w1 - w0)
ctx = [res[k][2]["prompt_tokens"] for k in range(N)]
print(f"decode: contexts {min(ctx)}-{max(ctx)} tokens ({sum(ctx)} in all); total {toks / wall:.1f} tok/s | steady (all {N} decoding) "
      f"{f'{steady:.1f} tok/s over {w1 - w0:.1f}s' if steady else 'no overlap'} | first token median {ttft[N // 2]:.2f}s, "
      f"last {ttft[-1]:.2f}s | wall {wall:.1f}s")
