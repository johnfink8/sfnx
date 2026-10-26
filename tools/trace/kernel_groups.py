"""A cupti_trace CSV's decode-window kernels (CUDA graph nodes) per window, by group, on each card: kernel time per
window, and the window's span on the card (first kernel start to last kernel end, per window, averaged) so idle gaps
show. Windows are told apart by gaps longer than GAP_US between graph kernels.

    kernel_groups.py TRACE.csv [GAP_US]"""
import collections
import sys

GROUPS = [
    ("ring sums", ("k_ring",)),
    ("experts (routed)", ("k_moe_",)),
    ("expert cache", ("k_cache_",)),
    ("router", ("k_router",)),
    ("dense q8 matrices", ("k_q8s",)),
    ("hc mixes", ("k_hc",)),
    ("attention", ("k_attn", "k_att_", "k_kv_append")),
    ("QSA indexer", ("k_idx_",)),
    ("recurrent core", ("k_gdn",)),
    ("PLE", ("k_ple",)),
    ("head / argmax", ("k_argmax", "k_head", "k_top")),
]


def group(name):
    for g, keys in GROUPS:
        if any(k in name for k in keys):
            return g
    return "other: " + name[:40]


path = sys.argv[1]
gap = int(sys.argv[2]) * 1000 if len(sys.argv) > 2 else 2_000_000
rows = collections.defaultdict(list)
with open(path) as f:
    next(f)
    for line in f:
        d, s, e, gs, name = line.rstrip("\n").split(",", 4)
        if gs.split(":")[0] == "0":
            continue
        rows[int(d)].append((int(s), int(e), name.split("(")[0].replace("void ", "").replace("fnx::", "")))
for dev in sorted(rows):
    ev = sorted(rows[dev])
    windows, cur = [], [ev[0]]
    for x in ev[1:]:
        if x[0] - max(y[1] for y in cur[-8:]) > gap:
            windows.append(cur)
            cur = []
        cur.append(x)
    windows.append(cur)
    windows = windows[1:-1]   # the first and last may be cut
    n = len(windows)
    tot = collections.Counter()
    span = 0
    for w in windows:
        span += max(y[1] for y in w) - w[0][0]
        for s, e, name in w:
            tot[group(name)] += e - s
    busy = sum(tot.values())
    print(f"card {dev}: {n} windows, span {span / n / 1e6:.2f} ms, kernel time {busy / n / 1e6:.2f} ms")
    for g, t in tot.most_common():
        print(f"  {t / n / 1e6:7.3f} ms  {100 * t / busy:5.1f}%  {g}")
