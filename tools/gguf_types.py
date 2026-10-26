"""Counts the ggml types of the tensors whose name matches a pattern, from a GGUF's header (first shard holds no
tensors of later shards: pass each shard).

    gguf_types.py PATTERN SHARD.gguf..."""
import collections
import re
import struct
import sys

NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
         13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
         21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 30: "BF16"}


def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode()


def skip_value(f, t):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in sizes:
        f.read(sizes[t])
    elif t == 8:
        read_str(f)
    elif t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            skip_value(f, et)
    else:
        raise SystemExit(f"unknown kv type {t}")


pat = re.compile(sys.argv[1])
counts = collections.Counter()
for path in sys.argv[2:]:
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF")
        _, n_t, n_kv = struct.unpack("<IQQ", f.read(20))
        for _ in range(n_kv):
            read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            skip_value(f, t)
        for _ in range(n_t):
            name = read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            f.read(8 * nd)
            (ty,) = struct.unpack("<I", f.read(4))
            f.read(8)
            if pat.search(name):
                counts[NAMES.get(ty, str(ty))] += 1
for k, v in counts.most_common():
    print(f"{v:4d} {k}")
