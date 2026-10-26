"""What a stretch of sfnx.log looked like from the engine: per admission the prompt, reused and read tokens and the
read time; per finished slot its tokens and time; how many slots were decoding when each event happened.

    panel_stats.py LOG_EXTRACT      (the sfnx: admitted / done / solo lines, in order)"""
import re
import statistics
import sys

ADM = re.compile(r"slot (\d+) \(lane \d+\) admitted: prompt (\d+) = (\d+) reused \+ (\d+) read in (\d+) ms; (\d+) slots decoding")
SOLO = re.compile(r"slot (\d+): prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms, (\d+) generated in (\d+) ms")
DONE = re.compile(r"slot (\d+) done: (\d+) tokens in (\d+) ms \((\w+)\), (\d+) slots decoding")
reads, busy, done, solo = [], [], [], []
for line in open(sys.argv[1]):
    if m := ADM.search(line):
        _, p, r, n, ms, b = map(int, m.groups())
        reads.append((p, r, n, ms))
        busy.append(b)
    elif m := SOLO.search(line):
        _, p, r, n, ms, g, gms = map(int, m.groups())
        reads.append((p, r, n, ms))
        solo.append((g, gms))
    elif m := DONE.search(line):
        _, t, ms, _, b = m.groups()
        done.append((int(t), int(ms)))
        busy.append(int(b) + 1)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


n = [r[2] for r in reads]
print(f"{len(reads)} prompts: median {statistics.median(r[0] for r in reads):,.0f} tokens, "
      f"{sum(r[1] for r in reads) / sum(r[0] for r in reads):.0%} of all prompt tokens reused")
print(f"  read per prompt: median {statistics.median(n):,.0f}, p90 {pct(n, 0.9):,}, max {max(n):,} tokens; "
      f"read time median {statistics.median(r[3] for r in reads):,.0f} ms, p90 {pct([r[3] for r in reads], 0.9):,} ms; "
      f"{sum(r[3] for r in reads) / 1000:.0f} s reading in all")
print(f"  reads over 10k tokens: {sum(x > 10000 for x in n)}; under 2k: {sum(x < 2000 for x in n)}")
if done:
    print(f"{len(done)} batched replies: median {statistics.median(d[0] for d in done):,.0f} tokens; "
          f"{sum(d[0] for d in done) / (sum(d[1] for d in done) / 1000):.1f} tok/s each on average")
if solo:
    print(f"{len(solo)} solo replies: {sum(g for g, _ in solo) / (sum(ms for _, ms in solo) / 1000):.1f} tok/s")
c = {k: busy.count(k) for k in sorted(set(busy))}
print("slots decoding at events: " + ", ".join(f"{k}: {v}" for k, v in c.items()))
