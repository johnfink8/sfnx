"""Pictures through serve/ against sFNX (greedy, thinking off): the logo (the Sphinx of Giza, ref/img/sphinx.png)
described; a second turn about it reads only the new turn (the picture stays cached); another picture (the old cat
logo) in the same place of an identical conversation is not mistaken for the first (the cache keys cells by picture);
a picture request and two text requests at once (the batch path). Each answer is checked for words only the right
picture gives. Exits non-zero on any failure.

    image_test.py [BASE_URL]        (default http://127.0.0.1:8095/v1)"""
import base64
import json
import os
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8095/v1"
HERE = os.path.dirname(os.path.abspath(__file__))
failures = []


def picture(name):
    with open(os.path.join(HERE, "..", "ref", "img", name), "rb") as f:
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(f.read()).decode()}}


def chat(messages, max_tokens=120):
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"], out["usage"], time.monotonic() - t0


def check(name, text, words):
    hit = [w for w in words if w.lower() in text.lower()]
    ok = bool(hit)
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {text[:110]!r}")
    if not ok:
        failures.append(name)


ask = "What does this picture show? Answer in two sentences."
first = [{"role": "user", "content": [picture("sphinx.png"), {"type": "text", "text": ask}]}]
a1, u1, s1 = chat(first)
check(f"sphinx ({u1['prompt_tokens']} prompt tokens, {s1:.1f} s)", a1, ["sphinx", "Sphinx"])
a2, u2, s2 = chat(first + [{"role": "assistant", "content": a1},
                           {"role": "user", "content": "What is the figure wearing on its head?"}])
check(f"second turn ({s2:.1f} s)", a2, ["headdress", "nemes", "headcloth", "head cloth"])
a3, u3, s3 = chat([{"role": "user", "content": [picture("cat.png"), {"type": "text", "text": ask}]}])
check(f"another picture, same place ({s3:.1f} s)", a3, ["cat", "Cat", "kitten"])
if any(w in a3.lower() for w in ("egypt", "pharaoh", "statue")):
    failures.append("another picture: the first picture's answer")
    print("FAIL another picture: answered about the first picture")

out = {}


def run(key, messages, words):
    try:
        out[key] = (chat(messages, 80)[0], words)
    except Exception as e:
        out[key] = (f"error {e}", words)


jobs = [("batch picture", first, ["sphinx", "Sphinx"]),
        ("batch text 1", [{"role": "user", "content": "Name the capital of France in one word."}], ["Paris"]),
        ("batch text 2", [{"role": "user", "content": "What is 12 times 12? Digits only."}], ["144"])]
threads = [threading.Thread(target=run, args=j) for j in jobs]
for t in threads:
    t.start()
    time.sleep(0.3)
for t in threads:
    t.join()
for key, _, _ in jobs:
    check(key, out[key][0], out[key][1])
print(f"{len(failures)} failed")
sys.exit(1 if failures else 0)
