"""The draft layer's K/V gaps after batched windows, through serve/ against sFNX with a tiny own pool: two conversations
decode together (batched windows: the draft layer does not read their tokens, so its K/V has gaps there), long enough to
run into pages the pool lent from expert slots; then conversation 0 goes on alone (a solo turn, whose drafts attend over
those gaps). Prints whether every request answered.

    gap_test.py [BASE_URL] [GEN]        (default http://127.0.0.1:8095/v1 3000)"""
import json
import random
import sys
import threading
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8095/v1"
GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 3000


def ask(messages, max_tokens):
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{BASE}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], d["usage"]


def opening(k):
    rng = random.Random(k)
    words = " ".join(rng.choice(["river", "stone", "lamp", "cloud", "pencil", "garden", "copper", "window", "violin",
                                 "harbor", "maple", "engine", "orbit", "canvas", "thunder"]) for _ in range(3000))
    return [{"role": "user", "content": f"Conversation {k}. Here is a list of words:\n{words}\n\nWrite the whole list back, word for word."}]


out, errs = {}, []


def one(k):
    try:
        out[k] = ask(opening(k), GEN)
    except Exception as e:  # noqa: BLE001 - reported below and fails the run
        errs.append(f"conversation {k}: {e!r}")


ts = [threading.Thread(target=one, args=(k,)) for k in range(2)]
for t in ts:
    t.start()
for t in ts:
    t.join()
if errs:
    raise SystemExit("batched phase failed:\n" + "\n".join(errs))
for k in range(2):
    print(f"conversation {k}: {out[k][1]['completion_tokens']} tokens together")
text, usage = ask(opening(0) + [{"role": "assistant", "content": out[0][0]},
                                {"role": "user", "content": "Now write the first 200 words of the list again."}], 400)
print(f"solo turn on conversation 0: prompt {usage['prompt_tokens']} tokens, {usage['completion_tokens']} out: {text[:60]!r}")
