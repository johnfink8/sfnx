"""A ref/*.bin prompt (token ids) back to text with the model's byte-level BPE vocabulary, for sending the same prompt
through an API.

    detok.py VOCAB.json PROMPT.bin OUT.txt      (the pack's tokenizer/vocab.json: token -> id)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
from refio import bpe_text, tokens  # noqa: E402

raw = bpe_text(sys.argv[1], tokens(sys.argv[2]))
open(sys.argv[3], "w").write(raw.decode("utf-8", errors="replace"))
print(f"{len(raw)} bytes")
