"""s27's first-window dump (S27_DUMP) against fnx_ref's records of the same names: per record, the cosine and the
relative RMS difference of the rows both hold (llama.cpp's rows are the whole prompt; s27's the first window's).

    s27_cmp.py REF.bin S27.bin"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
from refio import records  # noqa: E402

ref = {}
for name, a in records(sys.argv[1]):
    ref.setdefault(name, a)
for name, a in records(sys.argv[2]):
    if name not in ref:
        print(f"{name:24s} no reference")
        continue
    r = ref[name].reshape(-1, a.shape[-1])[: a.shape[0]]
    x = a[: r.shape[0]]
    cos = float((x * r).sum() / (np.linalg.norm(x) * np.linalg.norm(r) + 1e-30))
    rel = float(np.sqrt(((x - r) ** 2).mean() / ((r**2).mean() + 1e-30)))
    print(f"{name:24s} rows {x.shape[0]:3d}  cos {cos:.6f}  rel {rel:.2e}")
