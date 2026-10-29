"""Generated ids (one per line) back to text with the model vocabulary.

    s27_detok.py OUT.ids"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
from refio import bpe_text  # noqa: E402

ids = [int(x) for x in open(sys.argv[1]).read().split()]
sys.stdout.write(bpe_text(str(Path(__file__).resolve().parent.parent / "data/tokenizer/vocab.json"), ids).decode("utf-8", errors="replace"))
