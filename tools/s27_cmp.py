"""The 27B's per-layer dump (`bin/fnx_run MODEL check REF.bin OUT.bin [W]`) against ref/fnx_ref's of the same prompt
(its REGEX taking model.input_embed, linear_attn_out, attn_output, attn_residual, l_out and result_norm), layer by
layer, as the cosine and the relative RMS difference of the rows both hold:

    R0            model.input_embed
    attn_out-L    linear_attn_out-L (a recurrent layer) or attn_output-L (an attention layer): the mixer's sum
    ffn_out-L     l_out-L - attn_residual-L: the FFN's sum
    result_norm   result_norm

The dump's rows come from every window in order, the reference's are the prompt's first rows. Any record of the dump
without its reference, and any of R0, result_norm or a layer's two sums (every layer the reference has an l_out of)
missing from the dump, is an error (exit 1): a renamed or truncated record would otherwise drop out of the comparison
without a word.

    s27_cmp.py REF.bin OUT.bin"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
from refio import records  # noqa: E402

if len(sys.argv) != 3:
    raise SystemExit(__doc__)
ref = {}
for name, a in records(sys.argv[1]):
    rows = a.reshape(-1, a.shape[-1]).astype(np.float64)
    if name in ref:
        raise SystemExit(f"{sys.argv[1]}: duplicate reference record {name}")
    ref[name] = rows
got = {}
for name, a in records(sys.argv[2]):
    got.setdefault(name, []).append(a.reshape(-1, a.shape[-1]))
if not got:
    raise SystemExit(f"{sys.argv[2]}: no records")


def reference(name):
    """the reference rows for a dump record, or a string saying what is missing"""
    if name == "R0":
        return ref.get("model.input_embed", "model.input_embed")
    if name == "result_norm":
        return ref.get("result_norm", "result_norm")
    kind, l = name.rsplit("-", 1)
    if kind == "attn_out":
        r = ref.get(f"linear_attn_out-{l}", ref.get(f"attn_output-{l}"))
        return r if r is not None else f"linear_attn_out-{l} or attn_output-{l}"
    if kind == "ffn_out":
        lo, ar = ref.get(f"l_out-{l}"), ref.get(f"attn_residual-{l}")
        return lo - ar if lo is not None and ar is not None else f"l_out-{l} and attn_residual-{l}"
    return f"a reference for {name}"


def order(n):
    tail = n.rsplit("-", 1)[-1]
    return (n != "R0", int(tail) if tail.isdigit() else 1 << 30, n)


fails = []
layers = sorted(int(n.rsplit("-", 1)[1]) for n in ref if n.startswith("l_out-"))
if not layers:
    raise SystemExit(f"{sys.argv[1]}: no l_out records (fnx_ref's REGEX must take them)")
for want in ["R0", "result_norm"] + [f"{p}-{l}" for l in layers for p in ("attn_out", "ffn_out")]:
    if want not in got:
        fails.append(f"{want}: not in the dump")
for name in sorted(got, key=order):
    r = reference(name)
    if isinstance(r, str):
        fails.append(f"{name}: the reference has no {r}")
        continue
    x = np.concatenate(got[name], axis=0).astype(np.float64)
    if r.shape[1] != x.shape[1]:
        fails.append(f"{name}: width {x.shape[1]} against the reference's {r.shape[1]}")
        continue
    if len(x) != len(r):  # whole missing rows would otherwise compare the shared prefix and pass: count them
        fails.append(f"{name}: dump has {len(x)} rows against the reference's {len(r)}")
        continue
    n = len(x)
    cos = float((x * r).sum() / (np.linalg.norm(x) * np.linalg.norm(r) + 1e-30))
    rel = float(np.sqrt(((x - r) ** 2).mean() / ((r**2).mean() + 1e-30)))
    print(f"{name:24s} rows {n:5d}  cos {cos:.6f}  rel {rel:.2e}")
for f in fails:
    print(f"FAIL: {f}", file=sys.stderr)
sys.exit(1 if fails else 0)
