"""The draft layer's sources: each BF16 MTP tensor fetched from the checkpoint against the manifest's sha256 and
size, and the runtime files (rt) against a fresh build from the packed GGUF, byte for byte.

    mtp_check.py MTP_DIR RT_DIR [REBUILT_RT_DIR]"""
import hashlib
import json
import sys
from pathlib import Path

mtp, rt = Path(sys.argv[1]), Path(sys.argv[2])
bad = 0
for t in json.loads((mtp / "mtp-manifest.json").read_text()):
    f = mtp / t["file"]
    h = hashlib.sha256()
    with f.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    ok = h.hexdigest() == t["sha256"] and f.stat().st_size == t["bytes"]
    bad += not ok
    if not ok:
        print(f"MISMATCH {t['name']}")
print(f"checkpoint tensors: {bad} of the manifest's mismatch")
if len(sys.argv) > 3:
    re = Path(sys.argv[3])
    for f in sorted(p.name for p in re.iterdir()):   # draft_vocab.bin is not mtp_rt.py's
        same = (rt / f).read_bytes() == (re / f).read_bytes()
        bad += not same
        print(f"rt/{f}: {'identical' if same else 'DIFFERS'} to the rebuild")
sys.exit(1 if bad else 0)
