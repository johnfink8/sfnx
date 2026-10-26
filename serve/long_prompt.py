"""One long prompt through an OpenAI-compatible endpoint: the user message of a chat-formatted prompt text (as
tools/detok.py writes it), thinking off, greedy, MAX tokens; prints the first token's wait and the decode speed after it.

    long_prompt.py BASE_URL PROMPT.txt [MAX]        (MAX default 512)"""
import json
import sys
import time
import urllib.request

base, path = sys.argv[1], sys.argv[2]
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 512
text = open(path).read()
user = text.split("<|im_start|>user\n", 1)[1].rsplit("<|im_end|>\n<|im_start|>assistant", 1)[0]
body = {"model": "x", "messages": [{"role": "user", "content": user}], "max_tokens": max_tokens, "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.monotonic()
first = None
usage = None
with urllib.request.urlopen(req, timeout=1800) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        ev = json.loads(line[5:])
        if ev.get("usage"):
            usage = ev["usage"]
        if ev.get("choices") and ev["choices"][0]["delta"].get("content") and first is None:
            first = time.monotonic() - t0
end = time.monotonic() - t0
n = usage["completion_tokens"]
print(f"prompt {usage['prompt_tokens']} tokens: first token {first:.2f} s; {n} tokens, "
      f"{(n - 1) / (end - first):.1f} tok/s after the first")
