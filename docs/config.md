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
| `serve/sfnx-2card.json` | two cards (`"gpu": [0, 1]`), for an engine built with `make FNX_NCARD=2`: prod's settings without vision, a 32 GiB conversation cache, port 8095. The engine puts as many experts in RAM as the cards need at start; fp16 sums between the cards, 1.25x over the exactness bar on two (`FNX_RING_F32=1`: fp32, within it and ~30% slower; [docs/benchmarks.md](benchmarks.md#two-cards-and-one-card)) |
| `serve/sfnx-4card.json` | four cards (`"gpu": [0, 1, 2, 3]`), for an engine built with `make FNX_NCARD=4`: prod's settings with vision on card 3, a 32 GiB conversation cache, port 8095. Untested on hardware ([Card count](#card-count)) |
| `serve/sfnx.service.in` | the systemd unit's template: `make install` fills in the checkout (`@ROOT@`) and whoever runs it (`@USER@`, `@GROUP@`) and writes `/etc/systemd/system/sfnx.service`. It overrides the config's port with `--port 8080` |
| `serve/run.sh [CONFIG]` | the same as a transient unit on the config's own port (8095 for `sfnx.json`), with `LimitMEMLOCK` raised |

Every path in these configs is relative to the checkout — `cwd` is `.`, the engine is `bin/sfnx`, the log is
`serve/sfnx.log` — so a second worktree serves from its own tree. The GGUFs come through a `models` link in the
checkout, pointing at the folder holding `UD-Q4_K_XL/`, `Q8_0/` and `mmproj-F16.gguf`.

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
| `FNX_RING_F32=0` or `1` | decode's sums between cards as fp16 pairs (0) or fp32 (1); the prompt path's stay fp16. Default 0 on every count: fp32 costs ~30% of one request's decode on two cards (lab: new code 141 → 98 tok/s, prose 113 → 90). fp32 is the more exact: on three cards it takes the code prompt's KL from 5.75e-4 to 4.38e-4, both within the bar; on two, fp16 pairs leave code and edit 1.25x over the exactness bar (code 7.48e-4 against 6.06e-4; fp32 5.73e-4), so `exact base` there passes only with 1. Four, six and eight are unmeasured; each hop rounds a partial sum to f16 again, so their KL with 0 is the first thing to check |

**The lab**

| Setting | Effect |
| --- | --- |
| `FNX_LAB_TIMEOUT=N` | a command's watchdog in seconds (default 600) |
| `FNX_LAB_GDB=1` | run the worker under gdb |

## Boot requirement

The ring writes between cards, so every card's IOMMU group must be an identity domain. The engine checks at startup and
refuses to run otherwise: *`the ring's writes between cards need identity domains; boot with iommu=pt`*.

Every card must also be a CUDA peer of every other, not only of its ring neighbours: the prompt path's attention
exchange copies into every card and sets flags there. The engine enables peer access for every pair at start and stops
with `peer access` if one is refused (`nvidia-smi topo -p2p r` shows which pairs can). On this box (two sockets, the
cards on separate root ports) every pair can, with `iommu=pt`.

## Where this API differs from Strata's

- `logit_bias` is refused — the engine does not offer it, so the front end refuses the request.
- Not there: disk sessions (`POST /slots/0?action=save|restore`), `POST /v1/vram`, image generation, the MCP
  endpoint, Prometheus metrics, load/unload and idle unloading, model aliases, CORS, the settings view (`/config`).
  `GET /metrics` is the Monitor tab's JSON.
- A request's `strata_prefix`, `strata_tune` and `experimental_speed_projection` are not read.
- An engine that crashes is started again by the next request; there is no other loading or unloading.
