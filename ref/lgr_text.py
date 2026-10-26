"""The tokens an .lgr file took, as text (the model's byte-level BPE vocabulary).

    lgr_text.py VOCAB.json FILE.lgr"""
import sys

from refio import bpe_text, load_lgr

_, steps = load_lgr(sys.argv[2])
print(bpe_text(sys.argv[1], [s[0] for s in steps]).decode("utf-8", errors="replace"))
