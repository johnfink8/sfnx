"""A run's expert tallies against a profile's ranking: with the coldest G GiB (by the profile, as the loader ranks them)
in RAM, the share of the run's decode picks on experts in RAM, and those picks per round (an upper bound on the
experts staged per round: a window's repeated picks of one expert are staged once).

    route_eval.py PROFILE.bin RUN.bin ROUNDS EXPERT_BYTES.txt G [G ...]"""
import re
import sys
from array import array

NL, NEXP = 48, 512


def load(path):
    a = array("I")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if len(a) != 2 * NL * NEXP:
        raise SystemExit(f"{path}: {len(a)} counts, not {2 * NL * NEXP}")
    return a


prof, run, rounds = load(sys.argv[1]), load(sys.argv[2]), int(sys.argv[3])
mib = [0.0] * NL
for line in open(sys.argv[4]):
    m = re.match(r"\s*(\d+) gate .* ([\d.]+) MiB an expert", line)
    if m:
        mib[int(m.group(1))] = float(m.group(2))
if not all(mib):
    raise SystemExit(f"{sys.argv[4]}: not every layer's expert size")
n = NL * NEXP
td = sum(prof[:n]) or 1
tp = sum(prof[n:]) or 1
score = [(prof[i] / td + prof[n + i] / tp) / mib[i // NEXP] for i in range(n)]
order = sorted(range(n), key=lambda i: score[i])   # Python's sort is stable, like the loader's stable_sort
rd = sum(run[:n])
for g in map(float, sys.argv[5:]):
    moved, cold = 0.0, 0
    for i in order:
        if moved >= g * 1024:
            break
        moved += mib[i // NEXP]
        cold += run[i]
    print(f"{g:5.1f} GiB in RAM: {cold / rd:6.2%} of the run's decode picks, {cold / rounds:6.1f} a round")
