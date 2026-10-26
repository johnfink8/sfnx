"""The best drafts a lane for batched windows, by content and lane count, without trying every count: for each content
and each lane count from 2 up, the fixed drafts a lane (FNX_BATCH_FIXED, draft layer on) walk down from the last lane
count's best (6, the solo cap, before 2 lanes) while tok/s improves, and stop at the first drop. One row a lane (lookup
mode, no drafts) is the baseline; a best below it means no drafts. One row a lane emits one token a lane a window
whatever the content, so the baseline is measured once per lane count (or given: base=2:155.5,3:200.5,...). Uses the
lab (started, any FNX_BATCH_DRAFTS).

    python3 tools/bdecode_walk.py [N] [base=L:TOK_S,...] [CONTENT=PROMPTS ...]
        (default N 384; prose, code, edit, edit+prose)"""
import re
import subprocess
import sys

args = sys.argv[1:]
N = int(args.pop(0)) if args and args[0].isdigit() else 384
BASE = {}
if args and args[0].startswith("base="):
    BASE = {int(l): float(v) for l, v in (p.split(":") for p in args.pop(0)[5:].split(","))}
CONTENTS = dict(a.split("=", 1) for a in args) or {
    "prose": "prose2", "code": "code1", "edit": "edit1", "edit+prose": "edit1,prose2"}
LANES = (2, 3, 4, 5)


def lab(cmd: str) -> str:
    r = subprocess.run(["tools/lab.sh", "cmd", cmd], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: lab {cmd!r} (exit {r.returncode}):\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return r.stdout


def tok_s(lanes: int, prompts: str, drafts: str, fixed: int) -> float:
    lab(f"set FNX_BATCH_DRAFTS={drafts}")
    lab(f"set FNX_BATCH_FIXED={fixed}")
    out = lab(f"bdecode {lanes} {N} {prompts}")
    m = re.search(r"^bdecode \d+ lanes: every lane in: ([0-9.]+) tok/s, ([0-9.]+) rows", out, re.M)
    if not m:
        raise SystemExit(f"FAILED: no bdecode line for {lanes} lanes, {prompts}:\n{out[-2000:]}")
    return float(m.group(1))


targets = {}
for name, prompts in CONTENTS.items():
    prev = 6
    for lanes in LANES:
        if lanes not in BASE:
            BASE[lanes] = tok_s(lanes, prompts, "lookup", 0)
        base = BASE[lanes]
        tried = {}
        best_k, best = None, -1.0
        for k in range(prev, -1, -1):
            tried[k] = tok_s(lanes, prompts, "1", k)
            if tried[k] > best:
                best_k, best = k, tried[k]
            else:
                break
        target = best_k if best > base else 0
        targets[(name, lanes)] = target
        walked = "  ".join(f"{k}: {v:.1f}" for k, v in tried.items())
        print(f"{name:10s} {lanes} lanes: one row a lane {base:.1f} tok/s; fixed drafts {walked}; target {target} "
              f"({max(best, base) / base * 100 - 100:+.0f}%)", flush=True)
        prev = max(best_k, 1)
lab("unset FNX_BATCH_FIXED")
lab("set FNX_BATCH_DRAFTS=1")
print("targets (drafts a lane):", ", ".join(f"{n} {l}: {t}" for (n, l), t in targets.items()))
