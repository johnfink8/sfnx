"""Time to first token on a warm turn: a conversation of ~PREFIX tokens is read once (cold), then each later turn
adds ~2k new tokens to it (a tool result) and is timed to its first streamed token. Engine-neutral (any
OpenAI-compatible endpoint); greedy, short replies, thinking off.

    warm_ttft.py [BASE_URL] [PREFIX_WORDS] [TURNS]       (default http://127.0.0.1:8095/v1, 30000, 4)"""
import json
import random
import statistics
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8095/v1"
PREFIX = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
TURNS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
WORDS = ["river", "stone", "lamp", "cloud", "pencil", "garden", "copper", "window", "violin", "harbor", "maple",
         "engine", "orbit", "canvas", "thunder", "silver", "meadow", "lantern", "anchor", "falcon"]
rng = random.Random(7)


def text(n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def turn(messages):
    body = {"model": "x", "messages": messages, "max_tokens": 16, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.monotonic()
    first, reply = None, ""
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            delta = json.loads(line[5:])["choices"][0]["delta"].get("content") or ""
            if delta and first is None:
                first = time.monotonic() - t0
            reply += delta
    return first, reply


messages = [{"role": "system", "content": "You are a terse assistant. Answer in a few words."},
            {"role": "user", "content": f"Notes:\n{text(PREFIX)}\n\nHow many distinct words do the notes use, roughly?"}]
cold, reply = turn(messages)
print(f"cold read of ~{PREFIX} words: first token {cold:.2f} s")
warm = []
for k in range(TURNS):
    messages += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": f"More notes ({k + 1}):\n{text(1500)}\n\nWhich word appears last?"}]
    t, reply = turn(messages)
    warm.append(t)
    print(f"warm turn {k + 1} (+~2k tokens): first token {t:.2f} s")
print(f"warm turns: median {statistics.median(warm):.2f} s, max {max(warm):.2f} s")
