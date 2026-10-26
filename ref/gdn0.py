"""Layer 0's recurrent stages, card 0's heads (key heads 0-5; value heads 0-5, 16-21, 32-37), fnx against llama.cpp.

    gdn0.py REF_GDN0.bin FNX.bin"""
import sys

import numpy as np

from refio import load, records

ref = load(sys.argv[1])
items = {}
for name, v in records(sys.argv[2]):
    items.setdefault(name, []).append(v)
fx = {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in items.items()}
K = list(range(6))
V = [s * 16 + k for s in range(3) for k in K]
HS = 128
qch = [k * HS + i for k in K for i in range(HS)]
kch = [2048 + k * HS + i for k in K for i in range(HS)]
vch = [4096 + h * HS + i for h in V for i in range(HS)]
vcols = [h * HS + i for h in V for i in range(HS)]


def rel_rows(a, b):
    return " ".join(f"{np.sqrt(((a[t] - b[t]) ** 2).sum() / max((b[t] ** 2).sum(), 1e-30)):.1e}" for t in range(a.shape[0]))


def show(name, a, b):
    print(f"{name:12s} {rel_rows(a, b)}")


T = fx["gdn_qkv-0"].shape[0]
qkv = ref["linear_attn_qkv_mixed-0"].reshape(T, -1)
show("qkv", fx["gdn_qkv-0"], qkv[:, qch + kch + vch])
conv = fx["gdn_conv-0"]
show("q (normed)", conv[:, :len(qch)], ref["q_conv_predelta-0"].reshape(T, -1)[:, qch])
show("k (normed)", conv[:, len(qch):2 * len(qch)], ref["k_conv_predelta-0"].reshape(T, -1)[:, [c - 2048 for c in kch]])
show("v", conv[:, 2 * len(qch):], ref["v_conv_predelta-0"].reshape(T, -1)[:, vcols])
show("z", fx["gdn_z-0"], ref["z-0"].reshape(T, -1)[:, vcols])
show("g", fx["gdn_g-0"], ref["gate-0"].reshape(T, -1)[:, V])
show("beta", fx["gdn_beta-0"], ref["beta_sigmoid-0"].reshape(T, -1)[:, V])
show("o (delta)", fx["gdn_o-0"], ref["attn_output-0"].reshape(T, -1)[:, vcols])
show("y (normed)", fx["gdn_y-0"], ref["final_output-0"].reshape(T, -1)[:, vcols])
