"""The draft layer's routed experts as Q4_0 or Q8_0 from the BF16 checkpoint tensors, for FNX_MTP_EXPERTS.

    mtp_experts.py q4_0|q8_0 TENSORS_DIR OUT

TENSORS_DIR: Strata's mtp_fetch.py output (mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}.bin, BF16,
[512][1280][2560] with gate rows 0..639 and up rows 640..1279, and [512][2560][640]).
OUT: gate | up | down, [512][640][2560], [512][640][2560], [512][2560][640], the layout the loader unpacks Strata's
Q2_0 into. Blocks as ggml's quantize_row_q4_0_ref / quantize_row_q8_0_ref."""
import sys
from pathlib import Path

import numpy as np

H, FF, NE = 2560, 640, 512
BLOCK_BYTES = {"q4_0": 18, "q8_0": 34}


def bf16(path: Path, shape: tuple) -> np.ndarray:
    raw = np.memmap(path, dtype=np.uint16, mode="r")
    if raw.size != np.prod(shape):
        raise SystemExit(f"{path}: {raw.size} values, expected {shape}")
    return raw.reshape(shape)


def blocks(rows: np.ndarray) -> np.ndarray:
    return (rows.astype(np.uint32) << 16).view(np.float32).reshape(-1, 32)


def q4_0(rows: np.ndarray) -> bytes:
    """rows: (n, k) uint16 BF16 -> n * k / 32 blocks of 18 bytes (fp16 d, then 16 bytes: value i low, i + 16 high)."""
    x = blocks(rows)
    imax = np.argmax(np.abs(x), axis=1)   # the first of equal magnitudes, as the reference's strict <
    mx = x[np.arange(len(x)), imax]
    d = mx / np.float32(-8)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
    q = np.minimum(np.float32(15), np.trunc(x * idv[:, None] + np.float32(8.5))).astype(np.uint8)
    out = np.empty((len(x), 18), np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return out.tobytes()


def q8_0(rows: np.ndarray) -> bytes:
    """rows: (n, k) uint16 BF16 -> n * k / 32 blocks of 34 bytes (fp16 d = amax / 127, then 32 int8)."""
    x = blocks(rows)
    d = np.abs(x).max(axis=1) / np.float32(127)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
    v = x * idv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))).astype(np.int8)   # roundf: half away from zero
    out = np.empty((len(x), 34), np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in BLOCK_BYTES:
        raise SystemExit(__doc__)
    kind, src, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    quant = q4_0 if kind == "q4_0" else q8_0
    gu = bf16(src / "mtp.layers.0.mlp.experts.gate_up_proj.bin", (NE, 2 * FF, H))
    dn = bf16(src / "mtp.layers.0.mlp.experts.down_proj.bin", (NE, H, FF))
    tmp = out.with_suffix(out.suffix + ".part")
    with open(tmp, "wb") as f:
        for part in (slice(0, FF), slice(FF, 2 * FF)):
            for e in range(NE):
                f.write(quant(np.asarray(gu[e, part])))
        for e in range(NE):
            f.write(quant(np.asarray(dn[e])))
    want = 3 * NE * H * FF // 32 * BLOCK_BYTES[kind]
    if tmp.stat().st_size != want:
        raise SystemExit(f"{tmp}: {tmp.stat().st_size} bytes, expected {want}")
    tmp.rename(out)
    print(f"{out}: {want} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
