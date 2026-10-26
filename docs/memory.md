# Memory: the K/V arena and experts in RAM

Two problems with the same parts. Six conversations at 262,144 tokens need 1.57M positions of K/V, and the pool that
fits beside every expert holds ~450k (int8). And a Q8_0 build's routed experts do not fit in VRAM at all. Both are
solved by letting weights and K/V trade places, one expert slot at a time.

## The arena

The arena is on unless something turns it off: `serve/sfnx.json` sets nothing for it, and `serve/sfnx-q8.json` sets
`FNX_ARENA=0`. Rather than park or pause conversations when they outgrow the pool, the pool borrows expert memory.

- The 43 layers whose experts share the largest size keep their VRAM slots — 22,016 a card, 22.0 / 22.0 / 18.9 GiB — in
  one allocation with the K/V pool, right above the pool's own pages and ordered coldest first by a routing profile
  (`data/expert-routes.bin`).
- The pool's own pages are as many as the cards' free VRAM holds beside the slots and a reserve, sized after the
  engine's decode and prompt buffers and the draft layer are allocated. The reserve covers only what comes later — the
  window graphs, cuBLAS's workspaces — 512 MiB ([docs/config.md](config.md)).
- A card short of even the smallest own pool (another process holding VRAM on it) does not stop the engine: it puts
  that many slots' worth more of the coldest experts in RAM, places them again and says so
  (`fnx: VRAM short by N MiB ...`). Measured with 2 GiB held on card 0: 1,612 more experts in RAM, logits byte for byte
  the same.
- When a lane needs pages the free list lacks, the pool lends the next slots — at least 64 pages at a time, with every
  card idle for it, 0.1–1.3 ms. Their experts point back at their copies in page-locked RAM, and a window that picks
  one copies it into a slot first, as the Q8_0 build does. The pool can grow to ~2.9M positions.
- Slots come back once the top pages are free and no lend came in 30 s, refilled at once with the hottest experts that
  are out. While experts are lent, a lane idle for 120 s is parked to host memory to free its pages. With nothing lent,
  every expert is in a slot and the windows skip the cache's kernels.

### What it holds, and what it costs

| Check | Result |
| --- | --- |
| Logits against every expert in VRAM (`tools/arena_check.sh`), 62k prompt and its decode | byte for byte, with nothing lent and with a 64-page own pool that lends during the read and the decode |
| Batched windows while lending (`bin/batch_test`) | each lane its tokens alone, up to 718 slots lent on card 2 |
| Pool pressure through the front end, 461-page own pool | 9 of 9 conversations answer as alone; 6 lends, 2 returns; none parked, paused or waiting for a page |
| Six ~160k-token conversations (960k positions) decoding together (`serve/arena_long_check.sh`) | 237.6 tok/s combined (~40 a conversation); card 2 lent 5,201 slots (4.5 GiB of its experts), cards 0 and 1 2,464 — not re-measured with K/V by thirds |
| Cost with nothing lent (`tools/arena_speed.sh`), against every expert in VRAM | windows the same (5-row verify 17.88 / 17.85 ms, six lanes 21.11 / 21.09 ms); one request on new code the same (187.1 / 186.3 tok/s); on a code edit 4.1% slower (299.8 / 312.5) |
| The page-locked expert copies, with the cache off | cost a code edit 4.3%; batched windows nothing |
| Lent experts in a batch (six at once, coldest experts in RAM) | −3.9% at 8 GiB, −5.1% at 16 GiB |

The arena's own limits: a prompt read with experts lent reads those in place from RAM — the prompt path's stages are
sized from the experts out of a slot at load, and none are out with the arena — not measured yet. And requests read one
at a time without decoding (one token each) all land on the same lane, so each parks the one before: ~1 s each.

## Q8_0

The same model in Q8_0 (six files, 176 GiB on disk) does not fit in VRAM, so part of its routed experts stay in RAM. It
is not the served config; `serve/run.sh serve/sfnx-q8.json` runs it on port 8095.

**Experts in RAM.** `FNX_EXPERT_RAM_GIB=N` with `FNX_EXPERT_PROFILE=FILE` keeps the coldest N GiB of experts by the
profile in page-locked RAM. Every expert has a copy there; VRAM slots per expert size hold the ones in use, and a
decode window's experts not in a slot are copied in before the MoE runs. Profiles come from routing tallies written by
`FNX_ROUTE_COUNT=1` and `FNX_ROUTE_FILE`, decode and prompt picks weighted equally per byte.

On Q4_K_XL, output was identical to all experts in VRAM (KL 0), and decode on `ref/code1.bin` went from 176 to 160.6 /
155.5 / 146.7 tok/s with 8 / 16 / 24 GiB in RAM.

**K/V as int8.** `FNX_KV8=1` stores K/V rows as int8 (q8_0 groups; pooled QSA keys stay f16): 449,856 positions in the
f16 pool's bytes. KL against llama.cpp CUDA with f16 K/V, teacher-forced:

| Prompt | int8 K/V | f16 K/V | llama.cpp q8_0 K/V |
| --- | --- | --- | --- |
| `code1` | 7.1e-4 | 6.7e-4 | 5.1e-4 |
| `edit1` | 3.7e-6 | 5.1e-6 | 2.3e-6 |
| `long64` (62k tokens) | 3.10e-3 | 4.16e-3 | 4.98e-3 |

Five lanes at once run at the same speed either way: 229 against 230, and 197 against 199 tok/s.

**Prompts with experts in RAM.** `FNX_PF_PASS` chunks of 1,024 rows (default 4, at most 8) go through each layer before
the next, and the layer's experts not in a slot are copied into a VRAM stage — two of them, `FNX_PF_STAGE_MIB` each,
default 384 — while the layer before computes. Reads under `FNX_PF_STAGE_ROWS` tokens (default 2048) take the experts
in place. Q8_0, 52 GiB in RAM, int8 K/V, a 62,793-token prompt (`fnx_run`): 496 → 1,252 tok/s, logits identical.

**Prod's Q8 settings.** `serve/sfnx-q8.json`: 58 GiB of experts in RAM — 52 until windows of up to 16 rows took ~300
MiB more a card for the recurrent layers' snapshots, at which point 52 and 55 no longer fit a first long prompt's
buffers — int8 K/V, a 64 GiB conversation cache. Lab decode at 58 GiB with lookup drafts up to 15
(`tools/lab.sh decode`, greedy): code edit 137.8 tok/s (130.4 with lookup capped at 7), new code 111.5.

Speed and exactness against the reference: [docs/benchmarks.md](benchmarks.md), [docs/verification.md](verification.md).
