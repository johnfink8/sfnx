"""fnx dumps against the llama.cpp reference, layer by layer: relative RMS error of the stream mix input, the
mixer and FFN outputs, and the residual stack after each layer (fnx keeps the last write pending: applied here).

    compare.py REF.bin FNX.bin"""
import sys

import numpy as np

from refio import load, records

ref = load(sys.argv[1])


def cat(prefix, layer):
    """fnx records come per window: concatenate the rows of every window for this name"""
    rows = [v for k, v in fnx_items if k == f"{prefix}-{layer}"]
    return np.concatenate(rows, axis=0) if rows else None


fnx_items = list(records(sys.argv[2]))


def rel(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return np.sqrt(((a - b) ** 2).sum() / max((b ** 2).sum(), 1e-30))


print(f"{'layer':>5} {'hc_mixed':>9} {'mixer':>9} {'ffn_out':>9} {'R after':>9}")
for l in range(48):
    mix_ref = ref.get(f"hc_mixed-{l}")
    if mix_ref is None:
        mix_ref = ref.get(f"hc_mixed-{l} (reshaped)")
    mixer_ref = ref.get(f"linear_attn_out-{l}", ref.get(f"attn_output-{l}"))
    ffn_ref = ref.get(f"ffn_out-{l}")
    R_ref = ref.get(f"l_last-{l}", ref.get(f"l_last-{l} (reshaped)"))
    m, mo, fo = cat("hc_mixed", l), cat("mixer_out", l), cat("ffn_out", l)
    Rp, inj = cat("R_pending", l), cat("inj_ffn", l)
    R = None
    if Rp is not None and fo is not None and inj is not None:
        T = Rp.shape[0]
        w = 2.0 / (1.0 + np.exp(-inj / 4.0))   # [T, 4]
        R = Rp.reshape(T, 4, 2560) + fo.reshape(T, 1, 2560) * w.reshape(T, 4, 1)
    cols = []
    for a, b in ((m, mix_ref), (mo, mixer_ref), (fo, ffn_ref), (R, R_ref)):
        cols.append(f"{rel(a, b):9.2e}" if a is not None and b is not None and a.size == b.size else f"{'-':>9}")
    print(f"{l:5d} " + " ".join(cols))
