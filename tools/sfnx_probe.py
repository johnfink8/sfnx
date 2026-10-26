"""Drive an sfnx --serve process over its line protocol directly (no serve/): one GEN per step, printing the reply
lines' summary, the generated ids into OUT.json.

    sfnx_probe.py PROMPT.bin MAX_NEW OUT.json [CTX]

Steps: (1) the prompt; (2) the prompt + the reply + the reply's first 5 tokens again (the cached prefix must be
reused); (3) the first 2000 tokens of the prompt reversed (a different conversation: read from scratch).
Runs sfnx under systemd-run with the memlock limit raised; temperature 0."""
import json
import os
import struct
import subprocess
import sys
import time

FNX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = os.path.join(FNX, "models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")


def tokens(path):
    with open(path, "rb") as f:
        (nl,) = struct.unpack("<I", f.read(4))
        f.read(nl)
        (nd,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<q", f.read(8 * nd))
        return [int(x) for x in struct.unpack(f"<{n}f", f.read(4 * n))]


def main():
    prompt, max_new, out_path = tokens(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    ctx = sys.argv[4] if len(sys.argv) > 4 else "32768"
    cmd = ["sudo", "systemd-run", "--quiet", "--wait", "--pipe", "--collect", "-p", "LimitMEMLOCK=infinity",
           f"--uid={os.getuid()}", f"--gid={os.getgid()}", "-p", f"WorkingDirectory={FNX}", "--", f"{FNX}/bin/sfnx", "--serve",
           "--native", G, "--mtp", f"{FNX}/data/mtp-rt", "--draft-vocab", f"{FNX}/data/draft_vocab_en.bin",
           "--max-context", ctx, "--layer-split", "auto"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def line():
        s = p.stdout.readline()
        if not s:
            raise SystemExit(f"sfnx exited ({p.wait()})")
        return s.rstrip("\n")

    while True:
        s = line()
        print("<", s)
        if s.startswith("READY"):
            break

    def gen(ids, label):
        t0 = time.monotonic()
        p.stdin.write(f"GEN {max_new} " + ",".join(map(str, ids)) + "\n")
        out, first = [], None
        while True:
            s = line()
            if s.startswith("T "):
                if first is None:
                    first = time.monotonic() - t0
                out.append(int(s[2:]))
            elif s.startswith("DONE") or s.startswith("ERR"):
                print(f"{label}: {len(ids)} prompt tokens, first token after {first if first is None else round(first, 2)} s, "
                      f"{len(out)} tokens, {time.monotonic() - t0:.2f} s: {s}")
                if s.startswith("ERR"):
                    raise SystemExit(f"{label}: sfnx answered {s}")
                return out
            elif s.startswith("RESUME"):
                print(f"{label}: {s}")

    r1 = gen(prompt, "turn 1")
    r2 = gen(prompt + r1 + r1[:5], "turn 2 (extends turn 1)")
    r3 = gen(list(reversed(prompt[:2000])), "turn 3 (other conversation)")
    p.stdin.write("QUIT\n")
    p.stdin.close()
    if (rc := p.wait(timeout=120)) != 0:
        raise SystemExit(f"sfnx ended with exit status {rc}")
    with open(out_path, "w") as f:
        json.dump({"turn1": r1, "turn2": r2, "turn3": r3}, f)


main()
