"""The tensors whose type differs between two GGUF models (all shards), routed experts left out, by name pattern
(the layer number replaced by N) with the types in each and how many tensors have them.

    PYTHONPATH=llama.cpp/gguf-py .venv/bin/python tools/gguf_types_diff.py A-00001-of-N.gguf B-00001-of-M.gguf"""
import glob
import re
import sys
from collections import Counter

from gguf import GGUFReader


def types(first):
    out = {}
    for path in sorted(glob.glob(re.sub(r"-00001-of-(\d+)\.gguf$", r"-*-of-\1.gguf", first))):
        for t in GGUFReader(path).tensors:
            if not re.search(r"_exps\.weight$", t.name):
                out[t.name] = t.tensor_type.name
    return out


a, b = types(sys.argv[1]), types(sys.argv[2])
diff = Counter()
for name in sorted(set(a) | set(b)):
    if a.get(name) != b.get(name):
        diff[(re.sub(r"blk\.\d+\.", "blk.N.", name), a.get(name), b.get(name))] += 1
for (pat, ta, tb), n in sorted(diff.items()):
    print(f"{n:3d}  {pat:50s} {ta} -> {tb}")
