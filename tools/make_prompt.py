"""A prompt file for fnx_run (the "tokens" record of the reference dumps): a user message in Qwen's chat format, the
assistant turn opened with thinking off, tokenized with the model's tokenizer.json.

    make_prompt.py TOKENIZER.json OUT.bin "message"        (or - to read the message from stdin)

Needs the `tokenizers` package."""
import struct
import sys

from tokenizers import Tokenizer

tok_path, out_path, msg = sys.argv[1], sys.argv[2], sys.argv[3]
if msg == "-":
    msg = sys.stdin.read()
text = f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = Tokenizer.from_file(tok_path).encode(text, add_special_tokens=False).ids
name = b"tokens"
with open(out_path, "wb") as f:
    f.write(struct.pack("<I", len(name)) + name + struct.pack("<I", 1) + struct.pack("<q", len(ids)))
    f.write(struct.pack(f"<{len(ids)}f", *[float(i) for i in ids]))
print(f"{len(ids)} tokens -> {out_path}")
