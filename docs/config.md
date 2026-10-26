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
| `serve/sfnx.service.in` | the systemd unit's template: `make install` fills in the checkout (`@ROOT@`) and whoever runs it (`@USER@`, `@GROUP@`) and writes `/etc/systemd/system/sfnx.service`. It overrides the config's port with `--port 8080` |
| `serve/run.sh [CONFIG]` | the same as a transient unit on the config's own port (8095 for `sfnx.json`), with `LimitMEMLOCK` raised |

Every path in these configs is relative to the checkout — `cwd` is `.`, the engine is `bin/sfnx`, the log is
`serve/sfnx.log` — so a second worktree serves from its own tree. The GGUFs come through a `models` link in the
checkout, pointing at the folder holding `UD-Q4_K_XL/`, `Q8_0/` and `mmproj-F16.gguf`.

Engine arguments in the JSON: `--native` (the model's first shard), `--mtp` (the draft layer's runtime files),
`--draft-vocab`, `--max-context`, `--spec` (draft tokens a round), `--lookup` (the longest n-gram draft), `--batch`
(lanes), `--conversation-cache-mib`, `--vision`. `--layer-split` is accepted and ignored: the front end adds it for
three cards, and sFNX splits every layer across them instead.

## Engine settings

The ones that change results or are worth knowing. The rest are kernel-level tuning knobs — `grep -rn 'FNX_' src tools`
— and each throws on a value outside its range rather than clamping it.

**Experts and the arena**

| Setting | Effect |
| --- | --- |
| `FNX_ARENA=0` | turn the arena off: the pool is fixed and conversations park instead of borrowing expert memory. On unless set — prod sets nothing for it, `serve/sfnx-q8.json` sets 0 |
| `FNX_ARENA_RESERVE_MIB=N` | VRAM held back for the engine's other buffers. Default 1,792 at 1,024 prompt rows a chunk, +600 per further 1,024, +320 with batched drafts (a verify graph per row total): 3,912 at prod's 4,096 |
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
| `FNX_RING_F32=1` | the ring's sums as fp32 rather than fp16 pairs: KL 5.75e-4 → 4.38e-4 on the code prompt, and slower |

**The lab**

| Setting | Effect |
| --- | --- |
| `FNX_LAB_TIMEOUT=N` | a command's watchdog in seconds (default 600) |
| `FNX_LAB_GDB=1` | run the worker under gdb |

## Boot requirement

The ring writes between cards, so every card's IOMMU group must be an identity domain. The engine checks at startup and
refuses to run otherwise: *`the ring's writes between cards need identity domains; boot with iommu=pt`*.

## Where this API differs from Strata's

- `logit_bias` is refused — the engine does not offer it, so the front end refuses the request.
- Disk sessions (`POST /slots/0?action=save|restore`, the engine's SAVE/RESTORE) and `POST /v1/vram` (the engine's
  VRAM) are not supported: the front end refuses slot files with batch slots on, and the engine answers
  `ERR ... not supported by sfnx`.
- A request's `strata_prefix` — a pinned checkpoint at a shared prefix — is accepted and does nothing; the engine logs
  that once.
- Strata's per-request engine tuning (`strata_tune`) and `experimental_speed_projection` are ignored.
