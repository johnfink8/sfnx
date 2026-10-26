"""Where a card sits idle in a prompt read, from a cupti_trace CSV: inside the prompt path's span (as pf_kernels.py
takes it), each gap between one kernel's end and the next one's start, summed by the kernel before it (what the card
waited after), with the gaps' sizes in buckets.

    pf_gaps.py TRACE.csv [DEVICE] [TOP]"""
import collections
import os
import sys

path = sys.argv[1]
dev = int(sys.argv[2]) if len(sys.argv) > 2 else 0
top = int(sys.argv[3]) if len(sys.argv) > 3 else 15
rows = []
with open(path) as f:
    next(f)
    for line in f:
        d, s, e, _, name = line.rstrip("\n").split(",", 4)
        if int(d) == dev:
            rows.append((int(s), int(e), name))
rows.sort()
own = [r for r in rows if "k_hcb_prep" in r[2] or "k_pf_" in r[2]]
# READ=1 (a lab read: warm_graphs' windows, then the read): from the first prompt kernel after the last window
if os.environ.get("READ") == "1":
    last_win = max((r[1] for r in rows if "k_argmax_fin" in r[2]), default=0)
    own = [r for r in own if r[0] >= last_win]
t0, t1 = own[0][0], own[-1][1]
span = [r for r in rows if r[0] >= t0 and r[1] <= t1]


def short(n):
    n = n.split("(")[0]
    return n.replace("void ", "")[:60]


after = collections.defaultdict(lambda: [0.0, 0])
before = collections.defaultdict(lambda: [0.0, 0])
buckets = collections.Counter()
edges = [10, 50, 200, 1000, 5000, 1e18]   # us
total = 0.0
end = span[0][1]
prev = span[0][2]
for s, e, n in span[1:]:
    if s > end:
        g = (s - end) / 1000.0   # us
        total += g
        after[short(prev)][0] += g
        after[short(prev)][1] += 1
        before[short(n)][0] += g
        before[short(n)][1] += 1
        for b in edges:
            if g < b:
                buckets[b] += g
                break
    if e > end:
        end, prev = e, n
print(f"gpu{dev}: span {(t1 - t0) / 1e6:.1f} ms, idle {total / 1000:.1f} ms")
print("idle by gap size: " + ", ".join(f"<{b:g} us {buckets[b] / 1000:.1f} ms" for b in edges[:-1]) + f", more {buckets[1e18] / 1000:.1f} ms")
print("after:")
for k, (g, c) in sorted(after.items(), key=lambda kv: -kv[1][0])[:top]:
    print(f"  {g / 1000:8.1f} ms {c:6d}x  {k}")
print("before:")
for k, (g, c) in sorted(before.items(), key=lambda kv: -kv[1][0])[:top]:
    print(f"  {g / 1000:8.1f} ms {c:6d}x  {k}")
