"""Prints a GGUF's metadata keys matching a pattern (scalars and short arrays).

    gguf_kv.py PATTERN FILE.gguf"""
import re
import struct
import sys

SIZES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode(errors="replace")


def read_value(f, t):
    if t in SIZES:
        fmt = "<" + SIZES[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if t == 8:
        return read_str(f)
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [read_value(f, et) for _ in range(n)]
    raise SystemExit(f"unknown type {t}")


pat = re.compile(sys.argv[1])
with open(sys.argv[2], "rb") as f:
    if f.read(4) != b"GGUF":
        raise SystemExit("not a GGUF")
    _, _, n_kv = struct.unpack("<IQQ", f.read(20))
    for _ in range(n_kv):
        k = read_str(f)
        (t,) = struct.unpack("<I", f.read(4))
        v = read_value(f, t)
        if pat.search(k):
            if isinstance(v, list) and len(v) > 64:
                v = f"[{len(v)} values: {v[:8]} ...]"
            print(f"{k} = {v}")
