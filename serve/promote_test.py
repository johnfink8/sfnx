"""A request promoted from the solo path into a batch slot mid-reply (serve/ stops the solo request when a second one
arrives and continues it with BGEN) must give the same greedy text as alone. A: alone; then A again with B sent 1.5 s
after it (A is promoted); both at temperature 0.

    promote_test.py [BASE_URL]        (default http://127.0.0.1:8095/v1)"""
import json
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8095/v1"


def ask(text, out, key):
    body = {"model": "x", "max_tokens": 400, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out[key] = json.load(r)["choices"][0]["message"]["content"]


A = "Write a detailed essay about the history of GPUs, at least 300 words."
B = "List twenty animals, one per line."
out = {}
ask(A, out, "alone")
t = threading.Thread(target=ask, args=(A, out, "promoted"))
t.start()
time.sleep(1.5)
ask(B, out, "b")
t.join()
same = out["alone"] == out["promoted"]
print(f"alone {len(out['alone'])} chars, promoted {len(out['promoted'])} chars: {'SAME' if same else 'DIFFERENT'}")
if not same:
    k = next((i for i in range(min(len(out['alone']), len(out['promoted']))) if out['alone'][i] != out['promoted'][i]), None)
    print(f"first difference at char {k}: alone {out['alone'][max(0, k - 40):k + 40]!r}\npromoted {out['promoted'][max(0, k - 40):k + 40]!r}")
sys.exit(0 if same else 1)
