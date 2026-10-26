"""Per-window kernel time of the decode windows (CUDA graph nodes) in two cupti_trace CSVs: for one card, each kernel
name's time per window in A and in B and the difference, largest differences first.

    graph_kernels.py A.csv B.csv WINDOWS [DEVICE] [TOP]"""
import collections
import sys


def load(path, dev):
    tot = collections.Counter()
    with open(path) as f:
        next(f)
        for line in f:
            d, s, e, gs, name = line.rstrip("\n").split(",", 4)
            if int(d) != dev or gs.split(":")[0] == "0":
                continue
            tot[name.split("(")[0][:60]] += int(e) - int(s)
    return tot


a_path, b_path, win = sys.argv[1], sys.argv[2], int(sys.argv[3])
dev = int(sys.argv[4]) if len(sys.argv) > 4 else 1
top = int(sys.argv[5]) if len(sys.argv) > 5 else 15
a, b = load(a_path, dev), load(b_path, dev)
print(f"gpu{dev}: per window A {sum(a.values()) / win / 1e3:.3f} ms, B {sum(b.values()) / win / 1e3:.3f} ms (kernel time, summed)")
names = sorted(set(a) | set(b), key=lambda n: -(b[n] - a[n]))
for n in names[:top]:
    print(f"{a[n] / win / 1e3:8.3f} {b[n] / win / 1e3:8.3f} {(b[n] - a[n]) / win / 1e3:+8.3f} ms  {n}")
