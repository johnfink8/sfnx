# Configuration

Two layers: the front end's JSON config names the engine, its arguments and its files; the engine reads the rest from
its environment. Prod's settings are the code's defaults — an env var exists to try a better one, and a new behaviour
lands off by default until the GPU checks pass ([docs/verification.md](verification.md)).

## Config files

| File | What it is |
| --- | --- |
| `serve/sfnx.json` | prod: UD-Q4_K_XL, 262,144 context, 6 lanes, int8 K/V and the arena, conversation cache 131072 MiB, vision on card 2. The unit runs it on port 8080 |
| `serve/sfnx-q8.json` | the Q8_0 build: 58 GiB of experts in RAM, int8 K/V, a 64 GiB conversation cache ([docs/memory.md](memory.md)) |
| `serve/sfnx-gap.json` | a check config: prod's 262k context with a 440-page own pool (`FNX_ARENA_BASE_PAGES`), so reads and decodes lend |
| `serve/sfnx-pool.json` | a check config: 32k context, `--spec 4`, `--lookup 7`, no env of its own — the arena is on, only the context is small |
| `serve/sfnx-2card.json` | two cards (`"gpu": [0, 1]`), for an engine built with `make FNX_NCARD=2`: prod's settings without vision, a 32 GiB conversation cache, port 8095. The engine puts as many experts in RAM as the cards need at start; fp16 sums between the cards, slightly over the exactness bar (code KL 7.5e-4 against 6.1e-4, edit 7.1e-6 against 5.2e-6; the same top-1 as fp32) on two (`FNX_RING_F32=1`: fp32, within it and ~30% slower; [docs/benchmarks.md](benchmarks.md#two-cards-and-one-card)) |
| `serve/sfnx-4card.json` | four cards (`"gpu": [0, 1, 2, 3]`), for an engine built with `make FNX_NCARD=4`: prod's settings with vision on card 3, a 32 GiB conversation cache, port 8095. Untested on hardware ([Card count](#card-count)) |
| `serve/sfnx.service.in` | the systemd unit's template: `make install` fills in the checkout (`@ROOT@`), the config (`@CONFIG@`: `serve/sfnx.json`, or `make install CONFIG=path`) and whoever runs it (`@USER@`, `@GROUP@`) and writes `/etc/systemd/system/sfnx.service`. It overrides the config's port with `--port 8080` |
| `serve/run.sh [CONFIG]` | the same as a transient unit on the config's own port (8095 for `sfnx.json`), with `LimitMEMLOCK` raised |

Every path in these configs is relative to the checkout — `cwd` is `.`, the engine is `bin/sfnx`, the log is
`serve/sfnx.log` — so a second worktree serves from its own tree. The GGUFs come through a `models` link in the
checkout, pointing at the folder holding `UD-Q4_K_XL/`, `Q8_0/` and `mmproj-F16.gguf`.

**Your own config.** Keep it outside the checkout (`/etc/sfnx/sfnx.json` or `~/sfnx.json`, any path the service's
user can read), starting from a copy of `serve/sfnx.json`: its relative paths still resolve against the checkout,
wherever the file lives, so a pull never conflicts with it. `serve/run.sh /etc/sfnx/sfnx.json` tries it;
`make install CONFIG=/etc/sfnx/sfnx.json` makes it the service's from its next restart
(`sudo systemctl restart sfnx`), after checking it as a start would before touching the cards: its JSON and keys, the
sampling values, and the tokenizer, engine and vision files it names (`python -m front.check_config FILE` runs the same
check alone). A link stays a link, so repointing it switches configs at the next restart. The path may hold letters,
digits and `. _ / + -` only, as the unit's `ExecStart` takes it as is. The repository's configs are this
box's prod and the checks'.

Engine arguments in the JSON: `--native` (the model's first shard), `--mtp` (the draft layer's runtime files),
`--draft-vocab`, `--max-context`, `--spec` (draft tokens a round), `--lookup` (the longest n-gram draft), `--batch`
(lanes), `--conversation-cache-mib`, `--vision`.

The front end reads these keys and stops at start on any other, naming it: `exe`, `args`, `cwd`, `env`, `lib_dirs`,
`log`, `gpu` (the cards, as `CUDA_VISIBLE_DEVICES`), `parallel`, `engine_silence_s`, `vision`, `tokenizer`,
`model_name`, `host`, `port`, `api_key`, `allowed_hosts`, `api_monitor`, `sampling`, `reasoning_effort`,
`reasoning_budget_tokens`, `reasoning_close_retry`, `answer_room`, `fit_max_tokens`, `repeat_stop_tokens`,
`anthropic_thinking`. The `vision` section takes `exe`, `cpu_exe`, `mmproj`, `model`, `gpu`, `threads`, `max_tokens`,
`min_tokens`, `cuda_device`, and needs `--vision` in `args`. Unknown `sampling` keys stop the start too.

The front end's own environment (names kept from Strata): `STRATA_API_KEY` and `STRATA_ALLOWED_HOSTS` (as the config's
keys), `STRATA_DEBUG=1` (each reply's raw text in the journal; the unit sets it), `STRATA_LITERAL_STOP=0` (a stop token
quoted in a code span ends the reply after all), `STRATA_PARALLEL_SOLO=0` (a lane left alone stays in its batch slot),
`STRATA_KEEP_EMPTY_TURNS=1` (keep empty assistant turns in the prompt), `STRATA_ENGINE_READY_S`,
`STRATA_ENGINE_STALL_S`, `STRATA_VISION_READY_S`, `STRATA_VISION_ENCODE_S` (watchdogs, 0 = off),
`STRATA_MAX_BODY_MIB` (default 256), `STRATA_HTTP_BACKLOG` (default 256), `STRATA_BATCH_TRACE=1` (batch-slot trace).

## Card count

`FNX_NCARD` is a build setting: `make FNX_NCARD=n` builds for n cards (default 3, prod), and the engine refuses to start
unless exactly n cards are visible (the config's `"gpu"`). `build/ncard` records the count `build/` was made for; a
make with another count rebuilds everything. The Makefile and `src/card_split.hpp` accept 2, 3, 4, 6 and 8: the
counts that divide the 24 query heads and 48 recurrent value heads evenly, leave every card at least one 32-value block
of each expert's FF width, and need at most eight CUDA peers a card. Every layer splits by one rule
(`src/card_split.hpp`), which a static check holds to the two- and three-card splits the bases were made with:

| Cards | Recurrent heads a card (value / key) | Experts' FF slices | Query heads a card | Status |
| --- | --- | --- | --- | --- |
| 2 | 24 / 8 | 320 each | 12 | measured ([docs/benchmarks.md](benchmarks.md#two-cards-and-one-card)) |
| 3 | 16 / 6 (key heads 5 and 10 on two cards) | 224, 224, 192 | 8 | prod |
| 4 | 12 / 4 | 160 each | 6 | builds; untested on hardware |
| 6 | 8 / 3 or 4 (key heads 2, 5, 10 and 13 on two cards) | 128, 128, 96 x 4 | 4 | builds; untested on hardware |
| 8 | 6 / 2 | 96 x 4, 64 x 4 | 3 | builds; untested on hardware |

K/V is dealt by position (blocks of 4 positions, card j % n) and the vocabulary in ranges of ceil(vocab / n) tokens, the last card's
shorter, on every count.
Validating a count: [docs/verification.md](verification.md#other-card-counts).

## Engine settings

The ones that change results or are worth knowing. The rest are kernel-level tuning knobs — `grep -rn 'FNX_' src tools`
— and each throws on a value outside its range rather than clamping it.

**Experts and the arena**

| Setting | Effect |
| --- | --- |
| `FNX_ARENA=0` | turn the arena off: the pool is fixed and conversations park instead of borrowing expert memory. On unless set — prod sets nothing for it, `serve/sfnx-q8.json` sets 0 |
| `FNX_ARENA_RESERVE_MIB=N` | VRAM held back for what is allocated after the arena: the window graphs captured at first use, cuBLAS's workspaces, the lanes' page tables. The decode and prompt buffers and the draft layer are allocated before the arena is sized, so they need no share of it. Default 512: those took ~270 MiB a card on three cards and ~340 on two, through one to six lanes with batched drafts and a 114k prompt |
| `FNX_ARENA_BASE_PAGES=N` | the pool's own pages, fixed — for the checks |
| `FNX_EXPERT_RAM_GIB=N` | keep the coldest N GiB of experts in page-locked RAM. Needs `FNX_EXPERT_PROFILE` |
| `FNX_EXPERT_PROFILE=FILE` | the routing profile ranking expert coldness (`data/expert-routes.bin`) |
| `FNX_ROUTE_COUNT=1`, `FNX_ROUTE_FILE=FILE` | tally routing to write a profile: decode and prompt picks weighted equally per byte |

**K/V, drafts and batching**

| Setting | Effect |
| --- | --- |
| `FNX_KV8=1` | K/V rows as int8 (q8_0 groups; pooled QSA keys stay f16): ~450k positions in the f16 pool's bytes |
| `FNX_BATCH_DRAFTS=0\|lookup\|1` | what a batched window carries besides each lane's next token: one row a lane, lookup drafts only, or lookup plus the draft layer (default). The plan is `src/batch_plan.hpp` |
| `FNX_MTP_EXPERTS=q2_0` | the draft layer's experts as Strata's Q2_0 build instead of the Q4_0 made from the BF16 checkpoint: 59% of new code's drafts accepted against 63% |

**The prompt path**

| Setting | Effect |
| --- | --- |
| `FNX_PF_ROWS=N` | a prompt's rows per chunk: a multiple of 16 up to 8,192, default 4,096 (prod). ~20% faster than 1,024 past 1k tokens, for ~1.1 GiB more a card |
| `FNX_PF_PASS=N` | chunks of 1,024 rows through each layer before the next: default 4 with an expert cache, 2 without, at most 8 |
| `FNX_PF_STAGE_MIB=N` | each of the two VRAM stages that hold a layer's experts not in a slot while the layer before computes (default 384) |
| `FNX_PF_STAGE_ROWS=N` | reads shorter than this take the experts in place instead of staging them (default 2,048) |

**Sums between the cards**

| Setting | Effect |
| --- | --- |
| `FNX_RING_F32=0` or `1` | decode's sums between cards as fp16 pairs (0) or fp32 (1); the prompt path's stay fp16. Default 0 on every count: fp32 costs ~30% of one request's decode on two cards (lab: new code 141 → 98 tok/s, prose 113 → 90). fp32 is the more exact: on three cards it takes the code prompt's KL from 5.75e-4 to 4.38e-4, both within the bar; on two, fp16 pairs leave code and edit slightly over the exactness bar (code KL 7.5e-4 against 6.1e-4, fp32 5.7e-4; edit 7.1e-6 against 5.2e-6, fp32 3.1e-6), with the same top-1 as fp32 on every prompt, so `exact base` there passes only with 1. Four, six and eight are unmeasured; each hop rounds a partial sum to f16 again, so their KL with 0 is the first thing to check |

**Context past the trained 262,144**

| Setting | Effect |
| --- | --- |
| `FNX_ROPE_SCALE=S` (`--rope-scale S` in a config's args) | YaRN over the trained length, as llama.cpp's: `--max-context` may reach the GGUF's `context_length` (262,144) × S, and the engine refuses more — and more than 262,144 unscaled. Off by default (1 is off too). It changes the rotary at every position, so it costs some quality on every request, short ones included ([docs/verification.md](verification.md#rope-scaling)): set it only when the length is needed. The K/V pool does not grow with it ([docs/memory.md](memory.md)) |
| `FNX_YARN_ORIG_CTX=N` | the trained length YaRN scales from in place of the GGUF's `context_length`; only with `FNX_ROPE_SCALE` |
| (the GGUF's own YaRN) | a model whose GGUF sets `rope.scaling.type` yarn runs scaled by its `factor` from its `original_context_length` with no setting, as llama.cpp runs it; `FNX_ROPE_SCALE` and `FNX_YARN_ORIG_CTX` may repeat those values and are refused if they differ (`FNX_YARN_ORIG_CTX` past the trained length is refused too). A rotary sFNX cannot follow is refused at load (`src/rope_gguf.hpp`, tested by `bin/gguf_test`): another scaling type with a factor that scales — a factor with no type included, which llama.cpp takes as linear — llama.cpp's other rope keys at values that are not its defaults, a section order other than tyxz, a `rope_freqs` table |

**The lab**

| Setting | Effect |
| --- | --- |
| `FNX_LAB_TIMEOUT=N` | a command's watchdog in seconds (default 600) |
| `FNX_LAB_GDB=1` | run the worker under gdb |

**Start-up checks**

| Setting | Effect |
| --- | --- |
| `FNX_SKIP_BAR1_CHECK=1` | do not refuse a card whose BAR1 is smaller than its memory ([Boot requirement](#boot-requirement)): for a card whose peer access needs no BAR1 that large |

## Boot requirement

The engine checks every card as it loads, before it places anything on one (`src/card_check.hpp`, tested by
`bin/card_check_test` against real cards' numbers and sysfs), so a card it cannot use stops the load in seconds,
naming the card, rather than after the weights are on the cards.

The ring writes between cards, so every card's IOMMU group must be an identity domain. The engine refuses to run
otherwise: *`the ring's writes between cards need identity domains; boot with iommu=pt`*.

Every card must also be a CUDA peer of every other, not only of its ring neighbours: the prompt path's attention
exchange copies into every card and sets flags there. The engine enables peer access for every pair at start and stops
with `peer access` if one is refused (`nvidia-smi topo -p2p r` shows which pairs can). On this box (two sockets, the
cards on separate root ports) every pair can, with `iommu=pt`.

CUDA allowing peer access does not prove a link carries data, so before the first window the engine writes 512 KiB
from each card into a buffer on every other card, by a kernel's stores as the ring makes them (timed) and once by the
copy engines as the prompt path copies (not timed), and watches for every word on the receiving card
(`src/peer_probe.cu`; first it checks on each card that its watcher can fail, against words nobody wrote). A pair whose
words do not all arrive within a second (lost, misplaced, or under 0.5 MB/s) stops the engine with a line naming the
pair and how it was written, and how to leave the card out. A link that wedges a card can still leave it hung or, as
one did at a tester's, reset the host; when the probe sees a card still running or faulted it prints that and ends
the process at once, as nothing on the card can be freed safely. Otherwise the engine prints one line with the kernel
stores' rates (*`fnx: kernel stores between the cards: N pairs, A to B GB/s (512 KiB a pair)`*) and a warning for each
pair under 1 GB/s: the engine still runs, the tokens the same but slower, and the warning names the link to check.
On prod's V100s (PCIe Gen3) the probe measures 5.4 to 7.7 GB/s a pair, 68–97 µs per 512 KiB: 31 lab starts on
2026-10-28, the same band before its waits were bounded (16, 5.4–7.7) and after (15, 5.7–7.7), and 6.8 to 7.3 at
prod's first start on the probe's build (`c486d82`). On a tester's RTX 3090s (Gen4) `tools/p2p_store.cu` measured
~47 µs, ~11 GB/s. What the probe proves is that the words arrive, at its own buffers: not the flag-after-data
ordering the ring also relies on, and not the ring's addresses, so a fault that depends on either can still pass.

Every card's BAR1 must cover all its memory: the ring's stores into a card go through it. The engine reads each card's
BAR1 from `/sys/bus/pci/devices/<bdf>/resource` (readable by anyone on a standard kernel; one that hides it stops the
load at that path) and refuses a smaller one, naming the card and both sizes. A V100-32GB's BAR1 is 32 GiB, all of
it, so prod's cards pass. `FNX_SKIP_BAR1_CHECK=1` skips this check alone, and the engine says so as it loads.

## Other GPUs

`ARCH` is a build setting like `FNX_NCARD`: the compute capability the kernels are built for, default 70 for prod's
V100s, and `build/arch` records the one `build/` was made for (another rebuilds everything).

| `ARCH` | Cards | Status |
| --- | --- | --- |
| 70 | V100 | prod |
| 86 | RTX 3090, A6000, A40 | run on RTX 3090s at a tester's; logits not yet checked ([docs/verification.md](verification.md#rtx-3090s)) |
| 80, 89, 90, 100, 120 | A100; RTX 4090, L40S, RTX 6000 Ada; H100; B200; RTX 5090, RTX PRO 6000 Blackwell | every object compiles (nvcc 12.9); never run |
| 75 | Turing | refused: the prompt path's recurrent kernel takes ~73 KB of shared memory a block, Turing gives 64 (and no Turing card has 80 SMs) |

What a card must be beyond its arch is checked when the engine starts, not at build. As the model loads, the engine
refuses a card that is not the build's arch exactly (the driver would otherwise compile the kernels' PTX for it at
load, unchecked) or has fewer SMs than the hc read keeps resident (`HC_READ_BLOCKS` in `src/model.hpp`, 80: an RTX
3080's 68, an RTX 4080's 76 and an A10's 72 are too few); before the first window it tries every link ([Boot
requirement](#boot-requirement)). On the arches that never ran, the one Volta-tuned instruction the experts use
(`mma.sync.m8n8k4`, `src/moe_mma.cuh`) compiles, and NVIDIA's PTX documentation says it runs on later arches,
possibly much slower: speed there is unknown until measured. llama.cpp's libraries are built for the same arch, and a
make of the encoder or the reference tools stops if `llama.cpp/build/bin/libggml-cuda.so` holds kernels for another
arch or llama.cpp was configured with another nvcc version. `CUDA=/opt/cuda-12.9` takes nvcc, CUPTI and llama.cpp's compiler from a toolkit
somewhere other than `/usr/local/cuda`; `build/cuda` records the nvcc and its version, so another rebuilds everything.

RTX 3090s have run sFNX at a tester's (four cards, PCIe Gen4, AMD EPYC; [docs/verification.md](verification.md#rtx-3090s)).
What they needed:

- **Peer access.** GeForce drivers refuse it between cards; the tester ran the open kernel modules with the community
  P2P patch (aikitoria's fork of open-gpu-kernel-modules, driver 595.84).
- **Resizable BAR at full size on every card.** The patch maps a pair only when both cards' BAR1 covers their memory.
  With a smaller one `cudaDeviceCanAccessPeer` still says 1 and `nvidia-smi topo -p2p` still says OK, but stores
  between the cards are lost: the ring hung, and a probe through such a mapping reset the tester's host. The engine's
  BAR1 check (above) refuses such a card. `nvidia-smi -q | grep -A3 "BAR1 Memory"` shows the size; if
  `lspci -vvv -s <bus>`'s Resizable BAR lists nothing as large as the memory, no BIOS setting fixes it: the card
  needs a ReBAR VBIOS from its maker (one of the tester's four did).
- **CUDA 12.9.** With 12.8 and driver 595 the engine hung in `cudaGraphInstantiate` at start.
- **Fewer experts in VRAM.** 24 GB cards hold less than 32 GB ones: `FNX_EXPERT_RAM_GIB` with
  `FNX_EXPERT_PROFILE` ([docs/memory.md](memory.md)); 28 GiB on three cards, 12 GiB on four at 262,144 context with
  `FNX_KV8=1`. More is not always better: a card needs at least 2 x 16 x 10 = 320 resident slots a pool, so 42 GiB in
  RAM on three cards was refused. At 131,072 context on three cards a window graph ran out of memory at its first
  capture; the tester fixed the K/V pool's own pages instead (`FNX_ARENA_BASE_PAGES=366`). The setting for that is a
  larger `FNX_ARENA_RESERVE_MIB` (512 was measured on V100s, not on 3090s); `FNX_ARENA_BASE_PAGES` is for the checks.

Such a box's config belongs outside the checkout (above: *Your own config*), with its `"gpu"` list and, for a
toolkit outside the library path, `"lib_dirs"`.

## Where this API differs from Strata's

- `logit_bias` is refused — the engine does not offer it, so the front end refuses the request.
- Not there: disk sessions (`POST /slots/0?action=save|restore`), `POST /v1/vram`, image generation, the MCP
  endpoint, Prometheus metrics, load/unload and idle unloading, model aliases, CORS, the settings view (`/config`).
  `GET /metrics` is the Monitor tab's JSON.
- A request's `strata_prefix`, `strata_tune` and `experimental_speed_projection` are not read.
- An engine that crashes is started again by the next request; there is no other loading or unloading.
