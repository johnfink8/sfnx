"""How long a decoding request stalls while another request's long prompt is read: request A streams a long answer;
once it is decoding, request B sends PROMPT.txt's user message (cold). A's gaps between streamed tokens during B's
read are what B's chunks cost it; the longest should be about one chunk (1,024 rows, ~0.5 s).

    stall_test.py BASE_URL PROMPT.txt"""
import json
import sys
import threading
import time
import urllib.request

BASE, PATH = sys.argv[1], sys.argv[2]


def stream(messages, max_tokens, times):
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data:") and line != "data: [DONE]":
                ev = json.loads(line[5:])
                if ev.get("choices") and ev["choices"][0]["delta"].get("content"):
                    times.append(time.monotonic())


a_times, b_times = [], []
a = threading.Thread(target=stream, args=([{"role": "user", "content": "Write a long story about a lighthouse keeper, "
                                                                      "at least 3000 words."}], 4000, a_times))
a.start()
while len(a_times) < 20:
    time.sleep(0.05)
text = open(PATH).read()
user = text.split("<|im_start|>user\n", 1)[1].rsplit("<|im_end|>\n<|im_start|>assistant", 1)[0]
t_b = time.monotonic()
stream([{"role": "user", "content": user}], 16, b_times)
t_first = b_times[0] if b_times else time.monotonic()
a.join()
during = [t for t in a_times if t_b <= t <= t_first]
gaps = [y - x for x, y in zip(during, during[1:])]
print(f"B's prompt read until its first token: {t_first - t_b:.1f} s; A streamed {len(during)} pieces meanwhile")
if gaps:
    gaps.sort()
    print(f"A's gaps meanwhile: median {gaps[len(gaps) // 2] * 1000:.0f} ms, max {gaps[-1] * 1000:.0f} ms")
