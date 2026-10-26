"""Expert caches replayed on a decode routing trace (FNX_ROUTE_TRACE): for VRAM holding a share of the 24,576 routed
experts, the experts a window has to bring from RAM (a layer's distinct picks not in VRAM), per window, with
  static   the hottest by a tallies file (FNX_ROUTE_COUNT; the loader's ranking), never changing
  oracle   the same with the trace's own tallies (the best a fixed set can do on it)
  lru      one LRU over every layer's experts; a window's experts from RAM take the least recently used slots
  slru     llama.cpp's --moe-cache-mib: probation and protected LRUs, a hit moves to protected (half the slots)
  lfu      Strata's adaptive tier, simplified: per-expert counts decayed by 0.7 every 4 windows; an expert from RAM
           takes the slot of the coldest resident one if its count is higher
Every adaptive cache starts from the static set. An expert brought from RAM is copied to VRAM either way (to run
it), so keeping it costs no extra copy.

    cache_sim.py TRACE.bin PROFILE.bin SHARE [SHARE ...]      (SHARE: VRAM's share of the experts, e.g. 0.58)"""
import sys
from array import array
from collections import OrderedDict

NL, NEXP = 48, 512
N = NL * NEXP


def windows(path):
    """Each window as a list of per-layer sets of global expert ids (layer * NEXP + expert)."""
    t = array("I")
    with open(path, "rb") as f:
        t.frombytes(f.read())
    out, cur, prev = [], {}, -1
    for v in t:
        l, e = v >> 16, v & 0xFFFF
        if l < prev:
            out.append(list(cur.values()))
            cur = {}
        cur.setdefault(l, set()).add(l * NEXP + e)
        prev = l
    if cur:
        out.append(list(cur.values()))
    return out


def tallies(path):
    a = array("I")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    td, tp = sum(a[:N]) or 1, sum(a[N:]) or 1
    return [a[i] / td + a[N + i] / tp for i in range(N)]


def hottest(score, k):
    return set(sorted(range(N), key=lambda i: -score[i])[:k])


def run_static(ws, res):
    return [sum(len(s - res) for s in w) for w in ws]


def run_lru(ws, res):
    c = OrderedDict((i, None) for i in res)
    cap, out = len(res), []
    for w in ws:
        miss = 0
        for s in w:
            for i in s:
                if i in c:
                    c.move_to_end(i)
            new = [i for i in s if i not in c]
            miss += len(new)
            for i in new:
                c[i] = None
            while len(c) > cap:
                v = next(iter(c))
                if v in s:   # never the layer's own picks: move them up and take the next
                    c.move_to_end(v)
                    continue
                del c[v]
        out.append(miss)
    return out


def run_slru(ws, res):
    cap = len(res)
    prot_cap = cap // 2
    prob, prot = OrderedDict((i, None) for i in res), OrderedDict()
    out = []
    for w in ws:
        miss = 0
        for s in w:
            for i in s:
                if i in prot:
                    prot.move_to_end(i)
                elif i in prob:
                    del prob[i]
                    prot[i] = None
                    while len(prot) > prot_cap:
                        v, _ = prot.popitem(last=False)
                        prob[v] = None
            new = [i for i in s if i not in prot and i not in prob]
            miss += len(new)
            for i in new:
                prob[i] = None
            while len(prob) + len(prot) > cap:
                src = prob if prob else prot
                v = next(iter(src))
                if v in s:
                    src.move_to_end(v)
                    continue
                del src[v]
        out.append(miss)
    return out


def run_lfu(ws, res, every=4, decay=0.7):
    cnt = [0.0] * N
    resident = set(res)
    out = []
    import heapq
    for wi, w in enumerate(ws):
        miss = 0
        for s in w:
            for i in s:
                cnt[i] += 1.0
            new = [i for i in s if i not in resident]
            miss += len(new)
            if new:
                # the coldest residents outside this layer's picks, against the newcomers by count
                cold = heapq.nsmallest(len(new), (j for j in resident if j not in s), key=lambda j: cnt[j])
                for i in sorted(new, key=lambda i: -cnt[i]):
                    if not cold or cnt[i] <= cnt[cold[0]]:
                        break
                    v = cold.pop(0)
                    resident.discard(v)
                    resident.add(i)
        out.append(miss)
        if (wi + 1) % every == 0:
            cnt = [c * decay for c in cnt]
    return out


ws = windows(sys.argv[1])
prof = tallies(sys.argv[2])
own = [0.0] * N
for w in ws:
    for s in w:
        for i in s:
            own[i] += 1.0
picks = sum(len(s) for w in ws for s in w)
print(f"{len(ws)} windows, {picks / len(ws):.0f} distinct experts a window on average")
for share in map(float, sys.argv[3:]):
    k = int(share * N)
    res = hottest(prof, k)
    line = [f"VRAM {share:.0%} ({k} experts):"]
    for name, f, r in (("static", run_static, res), ("oracle", run_static, hottest(own, k)), ("lru", run_lru, res),
                       ("slru", run_slru, res)):
        m = f(ws, r)
        line.append(f"{name} {sum(m) / len(ws):6.1f}")
    print(" ".join(line) + "  experts from RAM a window", flush=True)
