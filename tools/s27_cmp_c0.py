"""Card 0's recurrent projections (s27 dump records c0_qkvz-L, c0_ab-L) against llama.cpp's full ones on card 0's
channels: key heads 0..5 (q and k), value heads 0..5, 16..20, 32..36 (v and z), alpha and beta of those heads.

    s27_cmp_c0.py REF.bin S27.bin LAYER"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
from refio import load  # noqa: E402

ref, got, l = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
HS, NKH = 128, 16
kh = list(range(6))
vh = [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 32, 33, 34, 35, 36]
qch = [k * HS + i for k in kh for i in range(HS)]
kch = [NKH * HS + c for c in qch]
vch = [2 * NKH * HS + h * HS + i for h in vh for i in range(HS)]
zch = [h * HS + i for h in vh for i in range(HS)]


def cmp(name, x, r):
    cos = float((x * r).sum() / (np.linalg.norm(x) * np.linalg.norm(r) + 1e-30))
    rel = float(np.sqrt(((x - r) ** 2).mean() / ((r**2).mean() + 1e-30)))
    print(f"{name:8s} cos {cos:.6f} rel {rel:.2e}")


q = got[f"c0_qkvz-{l}"]
T = q.shape[0]
mixed = ref[f"linear_attn_qkv_mixed-{l}"].reshape(-1, 10240)[:T]
z = ref[f"z-{l}"].reshape(-1, 6144)[:T]
n = len(qch)
cmp("q", q[:, :n], mixed[:, qch])
cmp("k", q[:, n : 2 * n], mixed[:, kch])
cmp("v", q[:, 2 * n : 2 * n + len(vch)], mixed[:, vch])
cmp("z", q[:, 2 * n + len(vch) :], z[:, zch])
ab = got[f"c0_ab-{l}"]
cmp("alpha", ab[:, : len(vh)], ref[f"alpha-{l}"].reshape(-1, 48)[:T][:, vh])
cmp("beta", ab[:, len(vh) :], ref[f"beta-{l}"].reshape(-1, 48)[:T][:, vh])
