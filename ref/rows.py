"""Per-row relative error of one fnx dump against one reference tensor.

    rows.py REF.bin FNX.bin FNX_NAME REF_NAME"""
import sys

import numpy as np

from refio import load, records

ref = load(sys.argv[1])
items = list(records(sys.argv[2]))
a = np.concatenate([v for k, v in items if k == sys.argv[3]], axis=0).astype(np.float64)
b = ref[sys.argv[4]].reshape(a.shape[0], -1).astype(np.float64)
a = a.reshape(b.shape)
for t in range(a.shape[0]):
    e = np.sqrt(((a[t] - b[t]) ** 2).sum() / (b[t] ** 2).sum())
    print(f"row {t:2d}: rel err {e:.3e}  |ref| {np.sqrt((b[t] ** 2).mean()):.3e}  |fnx| {np.sqrt((a[t] ** 2).mean()):.3e}")
