"""What FNX_ROUTE_COUNT's tallies say about keeping only the hottest experts in VRAM: for each share of the routed
experts' bytes left in RAM, the share of decode and of prompt picks that would land on an expert in RAM, with the
experts ranked by their picks (decode picks, then prompt picks) per byte across every layer.

    route_cover.py ROUTES.bin [EXPERT_BYTES.txt]     (expert_bytes.py's output: per-layer bytes; default 2.93 MiB each)"""
import math
import re
import sys
from array import array

NL, NEXP = 48, 512
a = array("I")
with open(sys.argv[1], "rb") as f:
    a.frombytes(f.read())
if len(a) != 2 * NL * NEXP:
    raise SystemExit(f"{sys.argv[1]}: {len(a)} counts, not {2 * NL * NEXP}")
mib = [2.93] * NL
if len(sys.argv) > 2:
    for line in open(sys.argv[2]):
        m = re.match(r"\s*(\d+) gate .* ([\d.]+) MiB an expert", line)
        if m:
            mib[int(m.group(1))] = float(m.group(2))
dec = [a[l * NEXP + e] for l in range(NL) for e in range(NEXP)]
pro = [a[(NL + l) * NEXP + e] for l in range(NL) for e in range(NEXP)]
size = [mib[i // NEXP] for i in range(NL * NEXP)]
td, tp, tb = sum(dec), sum(pro), sum(size)
print(f"picks: decode {td:,} ({td // (NL * 10):,} tokens), prompt {tp:,} ({tp // (NL * 10):,} tokens); experts {tb / 1024:.1f} GiB")
never_d = sum(1 for x in dec if x == 0)
print(f"experts never picked in decode: {never_d:,} of {NL * NEXP:,}; in both: {sum(1 for d, p in zip(dec, pro) if d == 0 and p == 0):,}")
for key, name in ((dec, "decode"), (pro, "prompt")):
    order = sorted(range(NL * NEXP), key=lambda i: key[i] / size[i])   # coldest first
    print(f"ranked by {name} picks: GiB in RAM -> decode picks in RAM, prompt picks in RAM")
    acc_b = acc_d = acc_p = 0.0
    marks = [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48]
    mi = 0
    for i in order:
        acc_b += size[i] / 1024
        acc_d += dec[i]
        acc_p += pro[i]
        while mi < len(marks) and acc_b >= marks[mi]:
            print(f"  {marks[mi]:3d} GiB ({marks[mi] / (tb / 1024):5.1%}): decode {acc_d / td:6.2%}, prompt {acc_p / tp:6.2%}")
            mi += 1
# a 1,024-row prompt chunk: the RAM experts it touches (each expert's picks spread evenly over the prompt tokens, a
# Poisson count per chunk: bursty routing touches fewer) and their bytes per card (a third of each)
n_tok = tp / (NL * 10)
order = sorted(range(NL * NEXP), key=lambda i: (dec[i] + pro[i] * td / tp) / size[i])   # both kinds, weighted equally
print("ranked by both: GiB in RAM -> decode picks in RAM, prompt picks in RAM, a 1,024-row chunk's RAM experts (MiB per card)")
acc_b = acc_d = acc_p = acc_t = 0.0
marks = [4, 8, 12, 16, 20, 24, 30, 36]
mi = 0
for i in order:
    acc_b += size[i] / 1024
    acc_d += dec[i]
    acc_p += pro[i]
    acc_t += (1 - math.exp(-pro[i] * 1024 / n_tok)) * size[i] / 3
    while mi < len(marks) and acc_b >= marks[mi]:
        print(f"  {marks[mi]:3d} GiB: decode {acc_d / td:6.2%}, prompt {acc_p / tp:6.2%}, chunk {acc_t:7.0f} MiB a card")
        mi += 1
# per layer: how even the routing is (the share of picks the hottest 25% of experts take)
spread = []
for l in range(NL):
    c = sorted((dec[l * NEXP + e] for e in range(NEXP)), reverse=True)
    spread.append(sum(c[: NEXP // 4]) / max(1, sum(c)))
print("decode picks taken by each layer's hottest 128 experts: min {:.0%}, median {:.0%}, max {:.0%}".format(
    min(spread), sorted(spread)[NL // 2], max(spread)))
