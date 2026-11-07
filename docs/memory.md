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
  one copies it into a slot first, as the Q8_0 build does. The pool can grow to ~3.5M positions (3,550,464 free on three
  cards at 262k, vision on; the engine prints it at start).
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

## The n-gram table on disk

The model's n-gram table (`per_layer_token_embd`, read at layer 1) is the largest thing on the host: 320,001,536 rows,
28.8 GB in Q4_K_XL (IQ4_NL, 90-byte rows) and 54.4 GB in Q8_0 builds (170-byte rows, as Unsloth's UD-Q5_K_XL ships
it too). Each position reads 16 rows of it, hashed from its token and the two before. By default it is locked in RAM.
On a box whose RAM is short, `FNX_NGRAM=disk` leaves it on the disk ([docs/config.md](config.md)): the engine hashes a
window's rows, takes what it can from a row cache (`FNX_NGRAM_CACHE_MIB`, 256 by default) and reads the rest with
O_DIRECT, one 4 KiB page a row (two for a row that crosses one), from a pool of `FNX_NGRAM_THREADS` (32) that wakes
as many threads as a round has reads. The page cache would hold 4 KiB for each ~100-byte row, and rows land on pages
at random, so the reads go around it. The bytes are the same: the rows gathered are the mapping's byte for byte, and
the logits are RAM mode's byte for byte.

**What it frees.** With experts in RAM every expert has a pinned host copy (`FNX_EXPERT_RAM_GIB` or not), so the table
comes on top of them. Measured, the engine process at idle after load (`serve/sfnx.json` and `serve/sfnx-q8.json` as
they are, plus `FNX_NGRAM=disk`):

| | Locked | Resident | Of it pinned (the expert copies, the prompt path's buffers) |
| --- | --- | --- | --- |
| Q4_K_XL, `FNX_NGRAM=disk` | 0.63 GiB (`token_embd`) | 74.8 GiB | 72.4 GiB |
| Q8_0, table in RAM | 51.3 GiB | 175.0 GiB | 120.4 GiB |
| Q8_0, `FNX_NGRAM=disk` | 0.63 GiB | 122.8 GiB | 120.2 GiB |

The picture encoder (0.5 GiB) and the front end (0.2 GiB) come on top, and so does the conversation cache as it fills
(`--conversation-cache-mib`: 64 GiB in prod's config, which a box short of RAM sets lower). From the GGUFs' headers,
UD-Q5_K_XL pins 98.4 GB of experts and locks 55.1 GB of embeddings: 142.9 GiB against a 128 GiB box, 92.2 GiB with
the table on the disk.

**How the reads hide.** A new prompt is mostly rows it has not read before (the engine's hash over `ref/`: 41.8% of
long2's 218,576 lookups are distinct rows, 49.4% of long64's, 41.9% of long164's); a prompt that continues text read
before finds 40–50% of its new rows in the cache. So the reads are hidden, not avoided:

- *Prompts.* Each pass's rows for the next pass are read on a thread of their own from the start of the pass (the host
  spends ~70% of a pass enqueueing it). The passes start at one chunk and double to the full four, so each hides the
  next one's reads; only the first chunk's are read before anything runs. A ramp from 1,024 rows instead of 4,096 read
  0.35 s faster but moved long64's logits off RAM mode's, so the ramp starts at a whole chunk. RAM mode (prod's) runs
  the same thread over the mapping, with no ramp: the RAM column's 17.05 s below is with it, and its logits are main's
  byte for byte.
- *Decode.* The next window's rows are read ahead as its tokens become known: the last token as soon as the round
  commits, each draft as the draft layer makes it. The window is launched before the last of them arrive, and layer 1
  waits on the cards for the host's rows (a flag in mapped memory), so the last draft's read hides behind the launch
  and layer 0. The batched windows (`step`, `verify_rows`) read synchronously before they launch.

**What it costs**, on this box's NVMe (Kingston NV3, DRAM-less, ~76k random 4 KiB reads/s at 32 in flight):

| Q4_K_XL, the lab (`tools/lab.sh`), three cards | RAM | `FNX_NGRAM=disk` |
| --- | --- | --- |
| A cold 62,793-token read (`read 62793`, an empty row cache) | 17.05 s | 18.25 s (+7%) |
| The same read with its rows cached | 17.05 s | 17.1 s |
| Decode: edit1 / code1 greedy, prose2 sampled (`lab.sh decode`) | 378.8 / 203.5 / 139.4 tok/s | 373.8 / 200.4 / 138.2 (−1.3 / −1.5 / −0.9%) |
| Batched, 6 lanes with drafts (`lab.sh batch`) | 284.0 tok/s | 285.1 |

| Served (`serve/sfnx.json`, `serve/sfnx-q8.json`) | main | RAM | `FNX_NGRAM=disk` |
| --- | --- | --- | --- |
| Q4_K_XL: a cold 32,045-token prompt (`serve/prompt_read.py 1 30000`) | | 3,640 tok/s | 3,289 (−10%) |
| Q8_0: the same prompt | 1,556 tok/s | 1,562 | 1,364 (−12%) |
| Q8_0: prose sampled, two 1,024-token replies (unseeded) | 99.9, 103.9 tok/s | 103.1, 96.7 | 101.0, 99.6 |
| Q4_K_XL: a 400-token greedy reply | | the same bytes as disk's | the same bytes as RAM's |
| Q8_0: the same reply | the reference | the same bytes | the same bytes |

On the way there, each change measured on prose2 with Q4 (RAM 139.4 tok/s): the synchronous read 131.7; reading the
window's rows ahead 132.4; waking only as many read threads as reads 136.3; one read-ahead thread in place of one a
request 136.7; the layer-1 wait with the last draft read ahead 137.6. The cold read: 23.1 s with the next pass's rows
read after its enqueueing, 19.4 s on their own thread, 18.25 s with the ramp.

A spinning disk is refused at start. The start-up log measures the disk (`n-gram table: its disk, 46,000 reads/s ...,
p50 485 us, p99 10,472 us` here; the p99 is the NV3's first reads after idle). Below ~30,000 reads/s (a full pass's ~118,000 cold pages in its ~4.3 s) the prompt reads
stop hiding behind the cards and prompts slow down with them.

Speed and exactness against the reference: [docs/benchmarks.md](benchmarks.md), [docs/verification.md](verification.md).
