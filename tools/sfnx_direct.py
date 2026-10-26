"""The sfnx engine process alone, without serve/: started with serve/sfnx.json's args (or --batch N), one greedy GEN of
a ref/*.bin prompt, its tokens and DONE line; the engine's own log line (rounds, ms a round) is on stderr.

    sfnx_direct.py PROMPT.bin MAX_NEW [--batch N] [--out TOKENS.txt]"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("max_new", type=int)
ap.add_argument("--batch", type=int)
ap.add_argument("--out")
a = ap.parse_args()
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(root, "serve", "sfnx.json")))   # its paths are the checkout's
args = [x for x in cfg["args"] if x != "--vision"]
if a.batch is not None:
    i = args.index("--batch")
    args[i + 1] = str(a.batch)
with open(a.prompt, "rb") as f:
    (nl,) = struct.unpack("<I", f.read(4))
    f.read(nl)
    f.read(4)
    (ne,) = struct.unpack("<q", f.read(8))
    ids = [int(v) for v in struct.unpack(f"<{ne}f", f.read(4 * ne))]
p = subprocess.Popen([os.path.join(root, cfg["exe"]), "--serve"] + args, cwd=root, stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, text=True, bufsize=1)
for line in p.stdout:
    if line.startswith("READY"):
        break
t0 = time.monotonic()
p.stdin.write(f"GEN {a.max_new} {','.join(map(str, ids))}\n")
p.stdin.flush()
toks, first = [], None
for line in p.stdout:
    if line.startswith("T "):
        toks.append(int(line.split()[1]))
        if first is None:
            first = time.monotonic()
    elif line.startswith(("DONE", "ERR")):
        end = time.monotonic()
        print(line.strip())
        break
print(f"{len(toks)} tokens; first token {first - t0:.2f} s; {(len(toks) - 1) / (end - first):.1f} tok/s after it")
if a.out:
    open(a.out, "w").write(" ".join(map(str, toks)) + "\n")
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=60)
