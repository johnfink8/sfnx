"""A shorter prompt file from a make_prompt.py one: the user message cut to its first tokens so that the whole prompt
is N tokens, the chat tail (from the message's closing <|im_end|> on: the assistant turn opened, thinking off) kept.

    cut_prompt.py MODEL.gguf IN.bin OUT.bin N        (PYTHONPATH=llama.cpp/gguf-py, with numpy)"""
import struct
import sys
from pathlib import Path

from gguf import GGUFReader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
from refio import tokens  # noqa: E402

gguf_path, in_path, out_path, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
field = GGUFReader(gguf_path).fields["tokenizer.ggml.tokens"]
vocab = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
im_end = vocab.index("<|im_end|>")
ids = tokens(in_path)
cut = max(i for i, t in enumerate(ids) if t == im_end)
tail = ids[cut:]
if n <= len(tail) or n >= len(ids):
    raise SystemExit(f"N must be between {len(tail)} and {len(ids)}")
out = ids[: n - len(tail)] + tail
with open(out_path, "wb") as f:
    f.write(struct.pack("<I", 6) + b"tokens" + struct.pack("<I", 1) + struct.pack("<q", len(out)))
    f.write(struct.pack(f"<{len(out)}f", *[float(i) for i in out]))
print(f"{out_path}: {len(out)} tokens (message cut from {cut} to {n - len(tail)}, tail {len(tail)})")
