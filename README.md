# sFNX

<p align="start"><img src="front/web/sphynx.svg" width="96" alt="sFNX"></p>

**Qwen3.8-Flash-Next at 382 tokens a second on a code edit, for $1,800 of used GPUs.**

Three Tesla V100-32GB cards at ~$600 each, on PCIe Gen3 with no NVLink. Six conversations at once, each holding the
full 262,144-token context, served by sFNX's own OpenAI chat and responses, Anthropic messages, tools and picture
front end (`front/`). Unsloth's UD-Q4_K_XL in prod; Q8_0 runs too with part of its experts in RAM.

**Every layer is split across all three cards**, so one request reads weights from all three cards' memory at the same
time. Splitting by layers — llama.cpp's default — passes one small activation per card boundary and works over any
link, but for one request the cards take turns: two of three idle. Splitting every layer buys all three cards'
bandwidth and costs a sum across them twice per layer, which NCCL cannot pay for over PCIe. So sFNX does the sums
itself: GPU threads storing into the next card's memory inside the window's CUDA graph, around a ring. Engines that
split tensors assume NVLink, and split attention by heads, which needs 2, 4 or 8 cards — not 3.

| What it splits | How |
| --- | --- |
| Experts (512 a layer) | a third of each expert's FF width per card |
| Recurrent (Gated DeltaNet) layers | 16 value heads per card |
| Attention K/V | by position: blocks of 4 dealt round-robin, partial results merged |
| Output head | a third of the vocabulary per card |
| Copied on every card | hyper-connection mixes, router, embeddings |

## Speed

UD-Q4_K_XL at prod's config (`serve/sfnx.json`), one request through `front/` unless the row says otherwise. Every run
behind these numbers, with its method: [docs/benchmarks.md](docs/benchmarks.md).

| Workload | |
| --- | --- |
| Code edit, greedy | 367–382 tok/s |
| New code, greedy | 200–229 tok/s |
| Prose sampled at temperature 1.0 | 139 tok/s |
| Six conversations decoding at once (1k-word prompts) | 280–291 tok/s combined, ~47 each |
| The same six with ~160k tokens of context each | 237.6 tok/s combined, ~40 each |
| 114k-token prompt, cold read | 3,373 tok/s |
| Warm turn (+2k on a 30k conversation), first token | 0.95 s median |

A range is the spread across runs, not a confidence interval; the 382 in the headline is the top of three greedy
code-edit runs. The gap between the first row and the third is the drafts: one verify window costs about the same
whether it accepts one token or sixteen, and code drafts where prose does not.

## Fewer cards

Three cards are what sFNX is made for. It also builds for **two** (`make FNX_NCARD=2`, `serve/sfnx-2card.json`): each
layer split in halves, a third of the experts in RAM, chosen by the engine at start. Ahead of Strata on the same two
cards, by less than on three. Its sums between the cards are fp16 by default, as on three, which leaves the code and
edit prompts over the exactness bar (code 1.2x, edit 1.4x) on two; `FNX_RING_F32=1` sends fp32, within the bar but with a ~30% penalty on
one request's decode:

| One request, two cards | sFNX (fp16 sums) | sFNX, fp32 sums | Strata |
| --- | --- | --- | --- |
| Code edit | 209–211 tok/s | 153–154 | 131–135 |
| New code | 137–163 tok/s | 107–112 | 94–99 |
| Prose, prod's sampling | 109–119 tok/s | 90–94 | 65–68 |
| 114k-token prompt, cold read | 42.1 s | 42.7 s | 53.7 s |

For **one** card, use Strata: it computes part of the experts a card cannot hold on the CPU, and was 2–3x sFNX's
decode and 4–5x its prompt reading there. Both, with the method: [docs/benchmarks.md](docs/benchmarks.md#two-cards-and-one-card).

## More cards (untested on hardware)

It also builds for **four, six or eight** cards (`make FNX_NCARD=4`; `serve/sfnx-4card.json` for four): every layer in
quarters, sixths or eighths by the same rule as two and three. Those are the counts the model's shapes allow: the
attention's 24 query heads and the 48 recurrent value heads divide evenly, every card keeps at least one 32-value block
of each expert's FF width, and every card can be a CUDA peer of every other (eight peers a card at most). Five and seven
are refused at build time; twelve would leave some cards a single FF block and needs more peers than CUDA gives;
twenty-four would leave some cards none.

**Nobody has run these counts.** They compile, and two checks run on the CPU: the split rule reproduces the two- and
three-card splits exactly and covers every head and FF block on every supported count (static checks in
`src/card_split.hpp`), and the prompt path's ring plan matches the two- and three-card step sequences and completes
consecutive sums on every supported count (`bin/pf_ring_plan_test`). Decode's sums go between cards as fp16 pairs on every count;
`FNX_RING_F32=1` sends fp32, more exact and ~30% slower on one request ([docs/config.md](docs/config.md#card-count)). How to validate a count:
[docs/verification.md](docs/verification.md#other-card-counts).

## Running it

Needs CUDA 12.9, g++-14, llama.cpp at a37cc46 checked out (or linked) as `llama.cpp/` (`make LLAMA=path` for
elsewhere), the front end's venv (`.venv`, from `pyproject.toml`; see [front/README.md](front/README.md)), the draft
layer's files in `data/mtp-rt/`, and a `models` link to the folder holding the GGUFs — `UD-Q4_K_XL/`, `Q8_0/` and
`mmproj-F16.gguf`.
Every path in the configs and scripts is relative to the checkout. The cards' IOMMU groups must be identity domains:
the ring writes between cards, and the engine refuses to start otherwise.

```bash
ln -s /path/to/Qwen3.8-Flash-Next models
python3.14 -m venv .venv && .venv/bin/pip install .   # the front end's packages (pyproject.toml)
make -j32                    # bin/sfnx and the picture encoder; ~2 min from scratch on this box
make install                 # the systemd unit for this checkout and user: port 8080, enabled at boot
sudo systemctl start sfnx    # ~80 s to load; the API is on port 8080
```

`serve/run.sh` runs the same config as a transient unit on port 8095; `serve/run.sh serve/sfnx-q8.json` the Q8_0
build. For two cards: `make FNX_NCARD=2 -j32 bin/sfnx`, then `serve/run.sh serve/sfnx-2card.json`; for four,
`FNX_NCARD=4` and `serve/sfnx-4card.json` (a build refuses to start on any other number of cards). Engine settings,
and the files they read: [docs/config.md](docs/config.md).

## Layout

| | |
| --- | --- |
| `src/` | the engine: kernels (`.cuh`), `engine.cu`, `prefill.cu`, `model.cpp`, `session.cpp`, `batch.cpp` |
| `tools/` | the engine process, the lab harness, the checks, probes and benchmarks |
| `front/` | the Python serving layer and web UI (from Strata's `serve/`, MIT) |
| `serve/` | configs, the systemd unit's template, run scripts, served tests |
| `vision/` | the picture encoder, on llama.cpp's mtmd |
| `ref/`, `data/` | reference prompts and llama.cpp's tools; tokenizer, draft vocabulary, routing profile |

## Limits

- One model, one box: three `sm_70` cards (or two; four, six or eight build but are untested), no FP8 and no
  CUDA 13 — the kernels are written for this generation.
- Not tied to one quantization, but a GGUF has to keep its embeddings, shared expert and draft-layer projections at
  Q8_0, and its routed experts in a type the kernels have ([docs/design.md](docs/design.md)); anything else throws at
  load. Both quants above are held to the same exactness bar ([docs/verification.md](docs/verification.md)).
- VRAM is essentially full at idle: 238 MiB free on cards 0 and 1, 2,492 MiB on card 2 beside the picture encoder.
- Prompts are read one at a time; the other lanes keep decoding between a read's chunks.
- Six lanes of temperature-1.0 prose decode ~5% under one-row-a-lane batching (`FNX_BATCH_DRAFTS=0`).
- A prompt re-read on a lane that held another conversation can differ in its last bits (~1e-4 first-token KL), and
  greedy text can part at a near-tie. Not traced; two fresh engines agree exactly.

## Docs

| | |
| --- | --- |
| [docs/design.md](docs/design.md) | the split, the ring, decode, the prompt path, the conversation cache, pictures |
| [docs/memory.md](docs/memory.md) | the K/V arena and experts in RAM, and what Q8_0 costs |
| [docs/benchmarks.md](docs/benchmarks.md) | every speed measurement, with its method |
| [docs/verification.md](docs/verification.md) | exactness against llama.cpp, the test inventory, the lab |
| [docs/config.md](docs/config.md) | engine settings, config files, where this API differs from Strata's |
