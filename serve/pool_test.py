"""The K/V pool under pressure, through serve/ against sFNX started with a small context (serve/sfnx-pool.json: 32k
positions in the pool, 6 lanes). Six ~9k-token conversations at once need more pages than the pool holds: idle lanes
are parked to host memory and admissions wait for decoding slots to end. Every request must answer, and each
conversation's greedy answer must equal its answer alone (read from scratch, then from a parked state).

With `copy`, the conversations ask for the list back, so the replies run long and outgrow the pool after they were
admitted: decoding slots are paused (parked) for each other's pages and resumed later.

    pool_test.py [BASE_URL] [N WORDS GEN [copy]]        (default http://127.0.0.1:8095/v1 6 9000 48)"""
import json
import random
import sys
import threading
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8095/v1"
N, WORDS, GEN = (int(x) for x in sys.argv[2:5]) if len(sys.argv) > 4 else (6, 9000, 48)
COPY = len(sys.argv) > 5 and sys.argv[5] == "copy"


def prompt(k):
    rng = random.Random(k)
    words = " ".join(rng.choice(["river", "stone", "lamp", "cloud", "pencil", "garden", "copper", "window", "violin",
                                 "harbor", "maple", "engine", "orbit", "canvas", "thunder"]) for _ in range(WORDS))
    if COPY:
        return f"Conversation {k}. Here is a list of words:\n{words}\n\nWrite the whole list back, word for word, then say DONE."
    return f"Conversation {k}. Here is a list of words:\n{words}\n\nWhich word comes first in the list, and what is {k} times 7?"


def ask(k, out, errors, key):
    body = {"model": "x", "max_tokens": GEN, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt(k)}]}
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.load(r)
            out[key] = d["choices"][0]["message"]["content"]
            out[key + ("tokens",)] = d["usage"]["completion_tokens"]
    except urllib.error.HTTPError as e:
        errors[key] = f"HTTP {e.code}: {e.read()[:300]!r}"
    except Exception as e:  # noqa: BLE001 - a thread's failure is reported with the others below
        errors[key] = f"{type(e).__name__}: {e}"


out, errors = {}, {}
for k in range(N):   # alone, one after another
    ask(k, out, errors, ("alone", k))
threads = [threading.Thread(target=ask, args=(k, out, errors, ("together", k))) for k in range(N)]
for t in threads:
    t.start()
for t in threads:
    t.join()
ok = not errors
for key, why in sorted(errors.items()):
    print(f"conversation {key[1]} ({key[0]}): ERROR {why}")
for k in range(N):
    if ("alone", k) in errors or ("together", k) in errors:
        continue
    a, b = out[("alone", k)], out[("together", k)]
    same = a == b
    ok = ok and same
    print(f"conversation {k}: {'SAME' if same else 'DIFFERENT'} ({out[('alone', k, 'tokens')]} / {out[('together', k, 'tokens')]} "
          f"tokens)  alone {a[:60]!r}  together {b[:60]!r}")
print("every conversation answered as alone" if ok else "FAILED")
sys.exit(0 if ok else 1)
