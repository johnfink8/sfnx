"""Compares two .lgr logit files of the same token sequence (fnx_logref / fnx_run logits): per step the top-1 of each,
the reference's top-1 - top-2 margin where they disagree, KL(ref || other) over the union of both top-64s (each
side's probabilities from its own log-sum-exp; a token outside a side's top-64 gets that side's 64th logit, an upper
bound), the top-10 overlap, and the longest run of agreeing top-1s from the start. With MAX_KL and MIN_TOP1 (a
percentage) it exits 1 when the KL mean is above MAX_KL or top-1 agreement under MIN_TOP1, saying which; an empty
file or one without steps is an error either way.

    lgrcmp.py REF.lgr OTHER.lgr [MAX_KL MIN_TOP1]"""
import math
import sys

from refio import load_lgr

if len(sys.argv) not in (3, 5):
    raise SystemExit("usage: lgrcmp.py REF.lgr OTHER.lgr [MAX_KL MIN_TOP1]")
rp, ref = load_lgr(sys.argv[1])
op, oth = load_lgr(sys.argv[2])
if rp != op:
    raise SystemExit("different prompts")
n = min(len(ref), len(oth))
if n == 0:
    raise SystemExit("no steps to compare")
agree, run, run_open = 0, 0, True
kls, overlaps, dis = [], [], []
for i in range(n):
    rt, rl, ri, rv = ref[i]
    ot, ol, oi, ov = oth[i]
    if ref[i][0] != oth[i][0]:
        raise SystemExit(f"step {i}: the files took different tokens ({ref[i][0]} vs {oth[i][0]})")
    r1, o1 = ri[0], oi[0]
    if r1 == o1:
        agree += 1
        if run_open:
            run += 1
    else:
        run_open = False
        dis.append((i, r1, o1, rv[0] - rv[1], ov[0] - ov[1]))
    rmap, omap = dict(zip(ri, rv)), dict(zip(oi, ov))
    kl = 0.0
    for t in set(ri) | set(oi):
        p = math.exp(rmap.get(t, rv[-1]) - rl)
        q = math.exp(omap.get(t, ov[-1]) - ol)
        if p > 0 and q > 0:
            kl += p * math.log(p / q)
    kls.append(max(kl, 0.0))
    overlaps.append(len(set(ri[:10]) & set(oi[:10])))
kls_s = sorted(kls)
print(f"{n} steps: top-1 agrees at {agree} ({100.0 * agree / n:.2f}%); first {run} steps agree in a row")
print(f"KL(ref||other): mean {sum(kls) / n:.2e}, median {kls_s[n // 2]:.2e}, p99 {kls_s[min(n - 1, int(0.99 * n))]:.2e}, "
      f"max {kls_s[-1]:.2e}; top-10 overlap mean {sum(overlaps) / n:.2f} of 10")
for i, r1, o1, rm, om in dis[:20]:
    print(f"  step {i}: ref {r1} (margin {rm:.4f}), other {o1} (its margin {om:.4f})")
if len(dis) > 20:
    print(f"  ... {len(dis) - 20} more")
if len(sys.argv) == 5:
    max_kl, min_top1 = float(sys.argv[3]), float(sys.argv[4])
    why = []
    if sum(kls) / n > max_kl:
        why.append(f"KL mean {sum(kls) / n:.2e} over {max_kl:.2e}")
    if 100.0 * agree / n < min_top1:
        why.append(f"top-1 {100.0 * agree / n:.2f}% under {min_top1:.2f}%")
    if why:
        print("BREACH: " + "; ".join(why))
        sys.exit(1)
    print(f"within KL mean <= {max_kl:.2e}, top-1 >= {min_top1:.2f}%")
