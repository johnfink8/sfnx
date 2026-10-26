"""Cancels that arrive right behind their request: a client that disconnects before its first token makes the front
end send STOP (solo) or BSTOP (batched) just after the GEN/BGEN, while the engine may still be busy. Each case sends
such a request with a long max_tokens, then a short request, and checks that the cancelled one stopped after a few
tokens (the engine log's generated count) and that the short one was not held up behind it.

    cancel_test.py [BASE_URL] [ENGINE_LOG]      (default http://127.0.0.1:8095/v1, serve/sfnx.log)"""
import http.client
import json
import re
import sys
import threading
import time
import urllib.parse
from pathlib import Path

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8095/v1"
log = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).with_name("sfnx.log"))
u = urllib.parse.urlparse(base)
LONG, MAX_CANCELLED, MAX_WAIT_S = 3000, 64, 8.0


def body(text, max_tokens, stream=True):
    return json.dumps({"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                       "stream": stream, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})


def send_and_drop(text, delay):
    """sends a streamed request and closes the connection after `delay` seconds, before reading anything"""
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    c.request("POST", u.path + "/chat/completions", body(text, LONG), {"Content-Type": "application/json"})
    time.sleep(delay)
    c.sock.close()


def short(text):
    t0 = time.time()
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=300)
    c.request("POST", u.path + "/chat/completions", body(text, 16, stream=False), {"Content-Type": "application/json"})
    r = c.getresponse()
    out = json.loads(r.read())
    if r.status != 200:
        raise SystemExit(f"short request: HTTP {r.status}: {out}")
    return time.time() - t0


def generated_since(offset):
    """the engine log's per-request generated counts written after offset (solo lines and batched slot lines)"""
    text = log.read_bytes()[offset:].decode(errors="replace")
    solo = [int(m) for m in re.findall(r"(\d+) generated in", text)]
    slots = [(int(n), f) for n, f in re.findall(r"slot \d+ done: (\d+) tokens in \d+ ms \((\w+)\)", text)]
    return solo, slots


fails = []


def case(name, keep_busy):
    busy = None
    if keep_busy:   # a long request decoding, so the next ones go through the batched path
        busy = threading.Thread(target=send_and_drop, args=("Write a very long essay about the history of computing.", 20.0))
        busy.start()
        time.sleep(3.0)
    off = log.stat().st_size
    send_and_drop(f"[{name}] Write a very long essay about rivers, at least 3000 words.", 0.05)
    wait = short(f"[{name}] Say OK.")
    time.sleep(1.0)
    solo, slots = generated_since(off)
    counts = [n for n, _ in slots] if keep_busy else solo
    cancelled = max(counts) if counts else None
    ok = cancelled is not None and cancelled <= MAX_CANCELLED and wait <= MAX_WAIT_S
    print(f"{name}: cancelled request generated {cancelled} tokens (limit {MAX_CANCELLED}), the short one took {wait:.2f} s "
          f"(limit {MAX_WAIT_S}); engine lines: solo {solo}, slots {slots} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append(name)
    if busy:
        busy.join()


case("solo", keep_busy=False)
case("batched", keep_busy=True)
sys.exit(1 if fails else 0)
