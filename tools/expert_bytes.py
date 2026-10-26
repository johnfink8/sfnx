"""Per-layer routed-expert sizes of a GGUF model (all shards): each layer's gate/up/down types and one expert's bytes.

    python tools/expert_bytes.py FIRST_SHARD.gguf   (needs llama.cpp's gguf-py)"""
import glob
import re
import sys

from gguf import GGUFReader

first = sys.argv[1]
shards = sorted(glob.glob(re.sub(r"-00001-of-(\d+)\.gguf$", r"-*-of-\1.gguf", first)))
per = {}
for path in shards:
    for t in GGUFReader(path).tensors:
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", t.name)
        if m:
            n_exp = int(t.shape[-1])
            per.setdefault(int(m.group(1)), {})[m.group(2)] = (t.tensor_type.name, int(t.n_bytes) // n_exp, n_exp)
total = 0
for l in sorted(per):
    d = per[l]
    b = sum(v[1] for v in d.values())
    total += b * d["gate"][2]
    print(f"{l:2d} gate {d['gate'][0]:6s} up {d['up'][0]:6s} down {d['down'][0]:6s} {b / 2**20:6.2f} MiB an expert")
print(f"routed experts in all: {total / 2**30:.1f} GiB")
