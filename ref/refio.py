"""The repository's file formats, read in one place (tools/ imports this too):

- fnx_ref dumps and ref/*.bin prompts: records of (name, ne, float32 data), a prompt file being one record named
  "tokens"; records() yields them in order, load() keeps the last of each name, tokens() reads a prompt's ids.
  numpy shape = reversed ggml ne, so [..., ne0].
- .lgr logit files (fnx_logref, fnx_run logits): load_lgr().
- the model's byte-level BPE vocabulary (the pack's tokenizer/vocab.json): bpe_text() turns ids back to bytes.

    refio.py REF.bin            list the records and the greedy next token of the last row"""
import json
import struct
import sys

import numpy as np


def records(path):
    """(name, array) for every record of a dump, in file order (a name may repeat: chunks of the same rows)."""
    with open(path, "rb") as f:
        while hdr := f.read(4):
            (nl,) = struct.unpack("<I", hdr)
            name = f.read(nl).decode()
            (nd,) = struct.unpack("<I", f.read(4))
            ne = struct.unpack(f"<{nd}q", f.read(8 * nd))
            n = int(np.prod(ne))
            yield name, np.frombuffer(f.read(4 * n), dtype=np.float32).reshape(tuple(reversed(ne)))


def load(path):
    """{name: array}, the last record of each name."""
    return dict(records(path))


def tokens(path):
    """A prompt file's token ids: its first record, which must be "tokens" with one dimension."""
    with open(path, "rb") as f:
        (nl,) = struct.unpack("<I", f.read(4))
        name = f.read(nl).decode()
        (nd,) = struct.unpack("<I", f.read(4))
        if name != "tokens" or nd != 1:
            raise SystemExit(f"{path}: the first record is {name!r} with {nd} dimensions, not a prompt's 'tokens'")
        (n,) = struct.unpack("<q", f.read(8))
        return [int(v) for v in struct.unpack(f"<{n}f", f.read(4 * n))]


def load_lgr(path):
    """(prompt ids, steps): each step (taken, lse, top ids, their logits), the top k of the file's k."""
    with open(path, "rb") as f:
        if f.read(4) != b"LGR1":
            raise SystemExit(f"{path}: not an .lgr file")
        k, np_ = struct.unpack("<Ii", f.read(8))
        prompt = struct.unpack(f"<{np_}i", f.read(4 * np_))
        (ns,) = struct.unpack("<i", f.read(4))
        steps = []
        for _ in range(ns):
            taken, lse = struct.unpack("<if", f.read(8))
            pairs = struct.unpack(f"<{2 * k}i", f.read(8 * k))
            ids = pairs[0::2]
            logits = [struct.unpack("<f", struct.pack("<i", v))[0] for v in pairs[1::2]]
            steps.append((taken, lse, ids, logits))
        return prompt, steps


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def bpe_text(vocab_path, ids) -> bytes:
    """ids back to the bytes they stand for, by vocab.json (token -> id); a special token is written as itself."""
    id_to = {i: s for s, i in json.load(open(vocab_path)).items()}
    byte_of = _bytes_to_unicode()
    raw = bytearray()
    for t in ids:
        s = id_to[t]
        raw += bytes(byte_of[c] for c in s) if all(c in byte_of for c in s) else s.encode()
    return bytes(raw)


if __name__ == "__main__":
    r = load(sys.argv[1])
    names = list(r)
    print(f"{len(names)} records; tokens {r['tokens'].astype(int).tolist()}")
    seen = set()
    for n in names:
        base = n.rsplit("-", 1)[0] if n[-1].isdigit() and "-" in n else n
        if base in seen:
            continue
        seen.add(base)
        print(f"  {n:28s} {r[n].shape}")
    if "result_output" in r:
        lo = r["result_output"]
        print("greedy next token per row:", lo.reshape(-1, lo.shape[-1]).argmax(-1).tolist())
