"""Kernel totals of a prompt-path run on one card, from a cupti_trace CSV: the span from the first to the last of the
prompt path's own kernels (k_hcb_prep / k_pf_*), kernel time by name inside it.

    pf_kernels.py TRACE.csv [DEVICE] [TOP]"""
import collections
import os
import sys

path = sys.argv[1]
dev = int(sys.argv[2]) if len(sys.argv) > 2 else 1
top = int(sys.argv[3]) if len(sys.argv) > 3 else 22
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
if not own:
    sys.exit(f"{sys.argv[0]}: no prompt-path kernels (k_hcb_prep / k_pf_*) on device {dev} in {path}")
t0, t1 = own[0][0], own[-1][1]
span = [r for r in rows if t0 <= r[0] <= t1]
tot, cnt = collections.Counter(), collections.Counter()
for s, e, n in span:
    k = n.split("(")[0][:60]
    tot[k] += e - s
    cnt[k] += 1
print(f"gpu{dev}: prompt path span {(t1 - t0) / 1e6:.1f} ms, kernels busy {sum(tot.values()) / 1e6:.1f} ms, {len(span)} kernels")
for k, v in tot.most_common(top):
    print(f"{v / 1e6:8.1f} ms {cnt[k]:6d}x  {k}")
