"""Layer 0's ring sum in an s27 dump: the cards' partials (pre_ring0..2) summed against what the ring left on card 0
(linear_attn_out-0), and that sum against llama.cpp's.

    s27_ring_check.py REF.bin S27.bin"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
from refio import load  # noqa: E402

ref, got = load(sys.argv[1]), load(sys.argv[2])


def cmp(name, x, r):
    cos = float((x * r).sum() / (np.linalg.norm(x) * np.linalg.norm(r) + 1e-30))
    rel = float(np.sqrt(((x - r) ** 2).mean() / ((r**2).mean() + 1e-30)))
    print(f"{name:28s} cos {cos:.6f} rel {rel:.2e}")


s = got["pre_ring0"] + got["pre_ring1"] + got["pre_ring2"]
T = s.shape[0]
cmp("ring result vs sum", got["linear_attn_out-0"], s)
r = ref["linear_attn_out-0"].reshape(-1, s.shape[1])[:T]
cmp("sum vs llama.cpp", s, r)
for c in range(3):
    cmp(f"card {c} partial vs llama.cpp", got[f"pre_ring{c}"], r)
