"""No output from the wrong conversation: N conversations at once, each a few thousand words of its own filler with its
own secret word in the middle, asked for the word; then each conversation's second turn (its cached prefix, maybe
parked and restored on another lane) asked for the word again. Every answer must hold its own word and no other's.

    identity_test.py [BASE_URL] [N] [WORDS] [SEED]      (default http://127.0.0.1:8095/v1, 6, 3000, time-based)"""
import json
import random
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8095/v1"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
WORDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else time.time_ns())
FILLER = ["river", "stone", "lamp", "cloud", "pencil", "garden", "copper", "window", "violin", "harbor", "maple",
          "engine", "orbit", "canvas", "thunder", "silver", "meadow", "lantern", "anchor", "falcon"]
SECRETS = ["zebracorn", "quillfish", "mangrove", "tessellate", "obsidian", "kumquat", "periwinkle", "saffron",
           "tundra", "xylophone", "juniper", "marzipan"]


def chat(messages):
    body = {"model": "x", "messages": messages, "max_tokens": 24, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]["message"]["content"]


secrets = rng.sample(SECRETS, N)
convs = []
for k in range(N):
    words = [rng.choice(FILLER) for _ in range(WORDS)]
    words.insert(WORDS // 2, f"(the secret word is {secrets[k]})")
    convs.append([{"role": "user", "content": " ".join(words) + "\n\nWhat is the secret word in the text above? Answer with the word only."}])


def turn(k, out):
    try:
        out[k] = chat(convs[k])
    except Exception as e:
        out[k] = f"error {e}"


bad = 0
for t in (1, 2):
    out = {}
    threads = [threading.Thread(target=turn, args=(k, out)) for k in range(N)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for k in range(N):
        others = [s for j, s in enumerate(secrets) if j != k and s in out[k].lower()]
        ok = secrets[k] in out[k].lower() and not others
        bad += not ok
        if not ok:
            print(f"FAIL turn {t} conversation {k}: wanted {secrets[k]!r}, got {out[k]!r}")
        convs[k] = convs[k] + [{"role": "assistant", "content": out[k]},
                               {"role": "user", "content": "Say the secret word once more, the word only."}]
print(f"{2 * N - bad} of {2 * N} answers held their own word")
sys.exit(1 if bad else 0)
