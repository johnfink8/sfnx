"""A ref/*.bin prompt sent to an OpenAI-compatible server, streamed: prints wall-clock (time.time) marks for the
request, the first token and the end, so a GPU sampler's timestamps can be split into prompt reading and decode.

    strata_stream.py MODEL.gguf PROMPT.bin [URL] [MAX_TOKENS] [greedy|served]   (PYTHONPATH=llama.cpp/gguf-py, numpy)

greedy (the default) sends temperature 0; served sends no sampling fields, so the server's config sampling applies
(prod's: temperature 0.6, top_p 0.95, presence 1.0), as for pi."""
import json
import random
import struct
import sys
import time
import urllib.request

from gguf import GGUFReader

tok_path, bin_path = sys.argv[1], sys.argv[2]
url = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8080/v1/chat/completions"
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 512
sampling = sys.argv[5] if len(sys.argv) > 5 else "greedy"
if sampling not in ("greedy", "served"):
    raise SystemExit(f"sampling {sampling!r}: greedy or served")
with open(bin_path, "rb") as f:
    (nl,) = struct.unpack("<I", f.read(4))
    f.read(nl)
    (nd,) = struct.unpack("<I", f.read(4))
    (n,) = struct.unpack("<q", f.read(8 * nd))
    ids = [int(x) for x in struct.unpack(f"<{n}f", f.read(4 * n))]
field = GGUFReader(tok_path).fields["tokenizer.ggml.tokens"]
vocab = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
cs = bs[:]
k = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + k)
        k += 1
unbyte = {chr(c): b for b, c in zip(bs, cs)}
raw = bytearray()
for i in ids:
    t = vocab[i]
    raw += t.encode() if t.startswith("<|") and t.endswith("|>") else bytes(unbyte[ch] for ch in t)
text = raw.decode("utf-8")
head, tail = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"
if not text.startswith(head) or tail not in text:
    raise SystemExit(f"{bin_path}: not make_prompt.py's chat format")
msg = text[len(head):text.index(tail)]
body = {
    "model": "strata",
    "messages": [{"role": "user", "content": f"[{random.getrandbits(64):016x}] {msg}"}],
    "max_tokens": max_tokens,
    "stream": True,
    "stream_options": {"include_usage": True},
    "chat_template_kwargs": {"enable_thinking": False},
}
if sampling == "greedy":
    body["temperature"] = 0
req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.time()
first = None
usage = None
with urllib.request.urlopen(req, timeout=1800) as resp:
    for line in resp:
        line = line.strip()
        if not line.startswith(b"data:") or line == b"data: [DONE]":
            continue
        ev = json.loads(line[5:])
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            if first is None and (ch.get("delta") or {}).get("content"):
                first = time.time()
t1 = time.time()
if first is None or usage is None:
    raise SystemExit(f"no content or no usage in the stream (first={first}, usage={usage})")
out = usage["completion_tokens"]
print(f"MARK start={t0:.3f} first={first:.3f} end={t1:.3f}")
print(f"{usage['prompt_tokens']} prompt tokens, first token after {first - t0:.2f} s; "
      f"{out} tokens in {t1 - first:.2f} s ({(out - 1) / (t1 - first):.1f} tok/s)")
