"""Residual stack after each layer (l_last-L), relative error of A against B, where either file may hold the rows in
several chunks (records with the same name, concatenated in order).

    drift.py A.bin B.bin"""
import sys

import numpy as np

from refio import records


def load_cat(path):
    items = {}
    for name, v in records(path):
        items.setdefault(name, []).append(v.reshape(v.shape[0], -1))
    return {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in items.items()}


a, b = load_cat(sys.argv[1]), load_cat(sys.argv[2])
for l in range(48):
    k = f"l_last-{l}"
    if k in a and k in b and a[k].shape == b[k].shape:
        e = np.sqrt(((a[k] - b[k]) ** 2).sum() / (b[k] ** 2).sum())
        print(f"layer {l:2d}: {e:.2e}")
