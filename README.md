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
edit prompts slightly over the exactness bar (code KL 7.5e-4 against 6.1e-4, edit 7.1e-6 against 5.2e-6; the same top-1 as fp32) on two; `FNX_RING_F32=1` sends fp32, within the bar but with a ~30% penalty on
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

## Other GPUs (RTX 3090s at a tester's; the rest never run)

`make ARCH=86` builds for RTX 3090s (compute capability 8.6); the default, 70, is this box's V100s. `ARCH=80`, `89`,
`90`, `100` and `120` (A100, Ada, H100, Blackwell) build too, but nobody has run them; Turing (75) is refused, short of
the shared memory a prompt kernel takes ([docs/config.md](docs/config.md#other-gpus)). A tester runs three and four RTX
3090s on PCIe Gen4 with no NVLink and reports ~178 tok/s for one request on three, 152–200 on four, their own measurements on their own
prompts ([docs/verification.md](docs/verification.md#rtx-3090s)). `lab.sh exact` has not run there, so an sm_86
build's logits are not yet held to the bar.

Splitting every layer needs every card to write into every other, and GeForce drivers refuse that. The tester's box
runs a community patch of NVIDIA's open kernel modules that allows it, with Resizable BAR at full size on every card
(the patch needs it; one card needed its maker's ReBAR VBIOS), CUDA 12.9 (12.8 hung at start) and some experts in RAM
on 24 GB cards. Workstation and datacenter cards generally allow it with NVIDIA's own driver (`nvidia-smi topo -p2p r`
says for a given box). There is no fallback for a box without it: a layer split, as Strata does, runs anywhere.

The engine checks each card as it loads, before anything is placed on it, and refuses one that is not the build's arch,
has fewer SMs than the hc read keeps resident (80, `HC_READ_BLOCKS`), sits in a translated IOMMU domain or has a BAR1
smaller than its memory. Before the first window it writes across every link between the cards and stops, naming the
pair, if anything does not arrive; a link under 1 GB/s gets a warning. What a box needs and what each refusal says: [docs/config.md](docs/config.md#other-gpus).

## The 27B (dense)

`make FNX_MODEL=27b` builds sFNX for Qwen3.8-27B, the family's dense model, at Q8_0 (`serve/sfnx-27b.json`): the same
front end, lanes, drafts, conversation cache and pictures, from an engine of its own behind the same interface
([docs/config.md](docs/config.md#model)). Every layer splits across the three cards as Flash-Next's does, except
attention, which splits by heads: the 27B's 4 KV heads let each card keep the K/V of the heads it reads.

| Workload, one request through the lab's runs unless the row says otherwise | |
| --- | --- |
| Code edit, greedy | 318 tok/s |
| New code, greedy | 157 tok/s |
| Prose, prod's sampling | 114 tok/s |
| Six conversations decoding at once, with drafts | 224 tok/s combined (`bin/batch_test`, `ref/long2` and `ref/code1`) |
| Prompt reading: 8k / 32k / 62k / 114k tokens | 2,435 / 2,264 / 1,965 / 1,602 tok/s |

Its logits are held to the widest gap between llama.cpp's own ways of running the 27B (its CPU backend, its CUDA
with and without int8 products of quantized activations), a bar 4–30x tighter than Flash-Next's. Code, the edit and
the picture are within it, with top-1 agreement of 100% / 100% / 99.5%; the 62k prompt agrees on every top-1 and sits
at 1.27x its KL. llama.cpp decodes the 27B Q8_0 at ~24 tok/s on this box, one request. On two of these cards
(`make FNX_MODEL=27b FNX_NCARD=2`) sFNX decodes the code edit at 255, new code at 133 and prose at 88 tok/s;
1Cat-vLLM run on the same two cards with the same prompts (NVFP4, 31% fewer bytes, with its DFlash2 drafts) gives 252,
184 and 78, and reads a 62k prompt at 1,427 tok/s against sFNX's 1,511. Details:
[docs/benchmarks.md](docs/benchmarks.md#qwen38-27b), [docs/verification.md](docs/verification.md#the-27b).

## Running it

Needs CUDA 12.9 (`make CUDA=path` for a toolkit other than `/usr/local/cuda`), g++-14, llama.cpp at a37cc46 checked
out (or linked) as `llama.cpp/` (`make LLAMA=path` for elsewhere), the front end's venv (`.venv`, from `pyproject.toml`; see [front/README.md](front/README.md)), the draft
layer's files in `data/mtp-rt/`, and a `models` link to the folder holding the GGUFs — `UD-Q4_K_XL/`, `Q8_0/` and
`mmproj-F16.gguf`.
Every path in the configs and scripts is relative to the checkout. The cards' IOMMU groups must be identity domains
and every pair of cards must be CUDA peers: the ring writes between cards, and the engine checks the cards and their
links at start and refuses to run otherwise ([docs/config.md](docs/config.md#boot-requirement)).

```bash
ln -s /path/to/Qwen3.8-Flash-Next models
python3.14 -m venv .venv && .venv/bin/pip install .   # the front end's packages (pyproject.toml)
make -j32                    # bin/sfnx and the picture encoder; ~2 min from scratch on this box
make install                 # the systemd unit for this checkout and user: port 8080, enabled at boot
sudo systemctl start sfnx    # ~80 s to load; the API is on port 8080
```

`serve/run.sh` runs the same config as a transient unit on port 8095; `serve/run.sh serve/sfnx-q8.json` the Q8_0
build. For two cards: `make FNX_NCARD=2 -j32 bin/sfnx`, then `serve/run.sh serve/sfnx-2card.json`; for four,
`FNX_NCARD=4` and `serve/sfnx-4card.json` (a build refuses to start on any other number of cards). For RTX 3090s,
`make ARCH=86` (llama.cpp's build too: a make stops if `llama.cpp/build` was made for another arch). A config of your own lives outside the checkout:
`make install CONFIG=/etc/sfnx/sfnx.json` (or `~/sfnx.json`, any path) makes it the service's. Engine settings,
and the files they read: [docs/config.md](docs/config.md).

## Layout

| | |
| --- | --- |
| `src/` | the engine: kernels (`.cuh`), `engine_common.cu` (what both models share), each model's layers and prompt path (`engine.cu` and `prefill.cu` for Flash-Next, `s27_engine.cu` and `s27_prefill.cu` for the 27B), `model.cpp`, `session.cpp`, `batch.cpp`; the start-up checks (`card_check.hpp`, `peer_probe.cu`) |
| `tools/` | the engine process, the lab harness, the checks, probes and benchmarks |
| `front/` | the Python serving layer and web UI (from Strata's `serve/`, MIT) |
| `serve/` | configs, the systemd unit's template, run scripts, served tests |
| `vision/` | the picture encoder, on llama.cpp's mtmd |
| `ref/`, `data/` | reference prompts and llama.cpp's tools; tokenizer, draft vocabulary, routing profile |

## Limits

- Two models (Flash-Next; the 27B as a build of its own, at Q8_0 only, on two or three cards; four build and are
  untested), measured on one box: three `sm_70` cards (or two; four, six or eight build but are untested). RTX 3090s
  (`ARCH=86`) run at a tester's, not yet checked against llama.cpp; 80, 89, 90, 100 and 120 build but have never run,
  and Turing is refused. No FP8, and no CUDA 13 (it drops `sm_70`).
- Every pair of cards must be CUDA peers over PCIe: GeForce cards need a community driver patch and full Resizable BAR
  for that, and there is no fallback for a box without it.
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
