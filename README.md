
# sFNX

<p align="start"><img src="front/web/sphynx.svg" width="96" alt="sFNX"></p>

**Qwen3.8-Flash-Next at Q4, six conversations at full 262k context, on about $3,000 of hardware:** three used Tesla
V100-32GB cards (~$600 each at late-2026 prices) in a Dell R740 with 294 GiB of RAM. One RTX 3090 costs ~$1,800, one
RTX 4090 ~$3,500 and one RTX 5090 ~$5,000 at the same prices, about $10,000 as a machine with the 128 GB of RAM the
5090 numbers below were measured with. At short context this box, at a third of that, decodes within ~10% of a 5090
running the fastest engines on 4-bit weights, ahead of the same card on Strata and of every two-3090 rig on record,
and it serves six conversations at once where they serve one:

| Flash-Next on... | Quant | One request decoding | At ~100k context | Prompt read | At once |
| --- | --- | --- | --- | --- | --- |
| **This box, sFNX** (~$3,000) | **Q4 (UD-Q4_K_XL)** | **367–382 tok/s** code edit, **200–229** new code (greedy); 139 prose sampled at temperature 1.0 | 135 tok/s (114k) | 3,373–3,667 tok/s | **six at 262k: 280–304 tok/s** in all |
| RTX 5090 + 128 GB RAM, Basalt ([LocalLLaMA comparison](https://old.reddit.com/r/LocalLLaMA/comments/1xdlunp/i_ran_all_the_popular_qwen_38_fn_engines_on_a/), greedy) | NVFP4 (4-bit experts) | 354 structured, 188 conversation, 180 prose | 184 | **6,492–8,666** | one |
| RTX 5090, Infernix (same comparison) | NVFP4 | 349 structured, 170 conversation, 163 prose | 176 | 6,436–6,630 | one |
| RTX 5090, Strata (same comparison) | NVFP4 | 204 structured, 146 conversation, 123 prose | 121 | 3,530–4,245 | one |
| 3090 + 3090 Ti, Strata, stock power ([club-3090 #1694](https://github.com/noonghunna/club-3090/issues/1694)) | 3-bit (IQ3_S) | 146–160 code, 118–122 narrative | — | 1,189–2,235 (90k) | one, at 131k |
| 2× RTX 3090, custom vLLM ([model card](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE)) | W4A16 experts, MTP3 | 84–89 (64 GB RAM), up to 111.5 (128 GB) | — | 3,401–4,191 (131k) | one |
| One RTX 3090, Strata (the owner's desktop) | 2-bit (IQ2_XS) | 75–90 tok/s | — | 1,700–2,000 | one |
| This box, Strata (the previous prod engine, its best config) | Q4 (UD-Q4_K_XL) | 68 tok/s | 73 (114k) | 2,454–2,971 | six: 199 tok/s |

sFNX numbers are through its own OpenAI-compatible front end (the speed table below has the method for each); the
others are as their sources report them, on their own prompts. Where the box loses: prompts read ~1.8–2.6x slower than on
a 5090 with Basalt or Infernix, and at ~100k context their decode is 30–36% ahead. sFNX gets there by splitting every
layer of the model across all three cards, so a single request uses all three cards' memory bandwidth, with its own
ring between the cards in place of NVLink, which these PCIe cards do not have.

## What it is

sFNX ("sphynx") runs Qwen3.8-Flash-Next (Unsloth UD-Q4_K_XL, 104 GiB) on three Tesla V100-32GB cards and serves
it through its own front end (`front/`, copied from Strata's `serve/`; see `front/README.md`): OpenAI chat
completions and responses, Anthropic messages, tools, strict tools with a JSON format, reasoning on and off,
pictures. It is built for one box (`infer`: a Dell R740, three V100s on PCIe Gen3, no NVLink, 294 GiB RAM) and one
model; a Q8_0 build of the same model also runs, with some experts in RAM (see "Q8_0").

Every number below was measured on that box; each table says which build and config.

## How it splits the model

Strata (and llama.cpp's layer split) give each card a third of the layers: for one request the cards take turns.
sFNX gives every card a third of **every** layer, so all three work on every token:

- **Experts:** each of the 512 experts per layer has a third of its FF width on each card. All 24,576 experts are in
  VRAM until the conversations need their room for K/V (see "K/V room: the arena"). (Prod Strata keeps 94% of them in
  VRAM and computes the rest from RAM.)
- **Recurrent (Gated DeltaNet) layers:** 16 value heads per card.
- **Attention layers:** the K/V cache is split by position: blocks of 4 positions dealt to cards 0, 1, 2 in turn, a
  third each. Every card attends over its own positions; the partial results are merged. (Card 2 used to hold half,
  and its attention gated every prompt layer: thirds read 30k tokens 2.5% faster and give the arena room for 3.50M
  positions instead of 2.70M.)
- **Output head:** a third of the vocabulary per card.
- **Copied on every card:** the hyper-connection mixes, the router, the embeddings.
- **Between the cards:** two sums per layer, passed around a ring (fp16 pairs: a value past fp16's range is an error,
  never a silent overflow). Decode moves them with the GPU's own threads, prompts with the copy engines.

On top of that:

- **Decode:** the model's MTP draft layer drafts 4 tokens (its routed experts as Q4_0 made from the BF16 checkpoint;
  Strata's Q2_0 build, `FNX_MTP_EXPERTS=q2_0`, has 59% of new code's drafts accepted against 63%); n-gram lookup in the context drafts when the text repeats
  (code edits), as many tokens as the match is long (3 to 15). One verify window checks up to 16 rows. Temperature,
  top-p, top-k, min-p and the repeat,
  frequency and presence penalties are applied as Strata does; sampled drafts are accepted by rejection sampling, so
  the output distribution is the model's.
- **Prompts:** chunks of 4,096 rows; the experts on the tensor cores (f16 dequantization + `mma.m8n8k4`), dense
  matrices as f16 cuBLAS GEMMs.
- **Six conversations at once:** K/V as int8 in a paged pool (192 positions a page) shared by six lanes; the decoding
  lanes share each window, each with its own drafts as a plan sizes them by its acceptance and the window's cost; a
  prompt being read lets the others decode between its chunks. The pool
  grows into the coldest experts' VRAM when the conversations need it, so all six can hold their full 262,144-token
  context at once (see "K/V room: the arena").
- **Conversation cache:** checkpoints at message boundaries and before late notes (`keep=N`), and every 4,096 tokens
  inside a long first message (a system prompt changed near its end, as an edited AGENTS.md does, is read again only
  from the last one before the change); conversations that leave a lane, or would lose 2,048 tokens or more to the
  next prompt on it, are parked in host memory (128 GiB), least recently used out first.
- **Pictures:** `bin/sfnx-vision` (llama.cpp's mtmd and the model's mmproj) encodes them on GPU2 (the CPU if that
  card cannot take it). A picture's cells get the encoder's rows and llama.cpp's interleaved M-RoPE positions; the cache
  matches a picture by its hash.

## Why split every layer

Splitting by layers is the usual way to spread a model over several GPUs (llama.cpp's default, Strata, vLLM's
pipeline mode): it passes one small activation per token at each card boundary, so it works over any link, plain
PCIe or mixed cards included, and with several requests in flight each card can work on a different one (Strata's
`--batch-groups 3`). Its cost is a single request: the cards take turns, two of three idle at any moment.

Splitting every layer makes one request use all three cards' memory bandwidth, which is what decode is limited by.
Its cost is a sum across the cards twice per layer: 96 per window. With NCCL that is ~85–110 µs per small
all-reduce on this box, about 10 ms a token in sums alone. Serving engines that split tensors
(vLLM, SGLang, TensorRT-LLM) assume NVLink, and split attention by heads, which needs a head count the card count
divides (2, 4 or 8 cards, not 3). sFNX gets around both: its ring sums are GPU threads storing into the next card's
memory inside the window's CUDA graph (the ring kernels take ~4.5 ms of a 22.5 ms six-row window on card 0, waiting
for the slowest card included), and K/V is split by position, the recurrent layers by value heads, the experts by FF
width. That makes it specific to this model and this box.

## Against consumer cards, in detail

At late-2026 prices the box comes to about $3,000, its three V100-32GB cards ~$600 each. One RTX 3090 (24 GB) is about
$1,800, one RTX 4090 (24 GB) about $3,500 and one RTX 5090 (32 GB) about $5,000, each before a machine with enough RAM
around it (about $10,000 for a 5090 with 128 GB).

What 12–24 GB cards get out of this model, from others' measurements (quants and setups differ; none on our
prompts):

| Setup | Quant | Decode |
| --- | --- | --- |
| RTX 5070 12 GB + 64 GB RAM, Strata ([its README](https://github.com/Niko1221/Strata)) | Q2_0 / IQ3_S | 94 / 53 tok/s |
| RX 9070 XT 16 GB + 47 GB RAM, Strata (same) | Q2_0 | 60 tok/s |
| RTX 3090 24 GB, Strata (same) | Q2_0 | "should write about 100–140" (their estimate, not a measurement) |
| RTX 3090 + CMP 170HX, llama.cpp ([report](https://reddit.sentinel-team.org/posts/1w11cze/snapshots/2026-09-04T21%3A20%3A12.3645Z)) | — | 70–80 tok/s short context, 17–18 near 262k |
| RTX 4090 24 GB + 32 GB RAM, llama.cpp, experts of 39 layers on the CPU (same report) | UD-IQ1_M | 15–27 tok/s |
| 4× RTX 3090, llama.cpp ([post](https://ar.linkedin.com/in/gustavosied)) | — | 43 tok/s layer split, 16 tensor split |

This box, one request through `front/` (`tools/strata_stream.py`, greedy, 1,024 and 636 tokens, three runs each; Q4
as prod serves it, Q8_0 as `serve/sfnx-q8.json` runs it):

| | New code (`ref/code1`) | Code edit (`ref/edit1`) |
| --- | --- | --- |
| UD-Q4_K_XL, prod's config | 200.4–229.1 tok/s | 366.9–381.6 tok/s |
| Q8_0, 52 GiB of experts in RAM (2026-10-25) | 116.4–122.3 tok/s | 139.9–146.2 tok/s |

Six conversations decoding at once: 280–291 tok/s in all with batched drafts, 303–304 without (six requests of 1,000
random words, 1,000 tokens each, steady part; Q4, prod's config; temperature-1.0 prose, the one load where batched
drafts lose).

So for what one consumer card costs, the box runs Q4, or Q8, faster than consumer cards get at 1–3 bits, at the full
262k context, for six conversations at once. What it gives up: the cards are used datacenter V100s (`sm_70`, no CUDA
13, no FP8; capped at 200 W each, 110–150 W each sampled in the speed runs), which is why an engine like this had to be written for them.
Q4 on a single 3090 (most of its 104 GiB of experts read from RAM every token) has no measurement here.

## Running it

Needs CUDA 12.9, g++-14, llama.cpp at a37cc46 checked out (or linked) as `llama.cpp/` (`make LLAMA=path` for
elsewhere), the front end's venv (`.venv`, packages in `front/README.md`), and the draft layer's files in
`data/mtp-rt/` (Strata's `mtp_rt.py` output plus `tools/mtp_experts.py`'s experts from the BF16 checkpoint; see
`front/README.md`). Every path in the configs and scripts is the checkout's; the GGUFs are found through a `models`
link in it, to the folder holding `UD-Q4_K_XL/`, `Q8_0/` and `mmproj-F16.gguf`.

```bash
ln -s /path/to/Qwen3.8-Flash-Next models
make -j32                    # bin/sfnx and the picture encoder (and llama.cpp's libraries, first time); kernels.cu
                             # compiles as 9 parts in parallel: ~2 min from scratch on this box
make install                 # the systemd unit for this checkout and user: port 8080, enabled at boot
sudo systemctl start sfnx    # ~80 s to load; clients use http://infer:8080/v1
```

`serve/run.sh` runs the same config as a test unit on port 8095.

## Speed against prod Strata

sFNX as prod runs it (`serve/sfnx.json`: int8 K/V and the arena), through its own front end; Strata as this box's
service ran it before the switch (the same model file, `--batch 6 --batch-groups 3`, MTP drafts, 94% of the experts
in VRAM). GPU utilization is the mean over the run of all three cards.

| Workload | sFNX | Prod Strata |
| --- | --- | --- |
| A code-review panel's first round, replayed (5 reviewers, 83 calls, prompts up to ~48k tokens, the recorded replies' lengths, two runs) | **264–271 s, 154.5–168.5 tok/s** over the run, first token median 1.0 s, GPUs 90% | 484 s, 76.5 tok/s |
| A pi reviewer session, replayed (code-style reviewer's first round, 14 calls; its first call an 11k real prompt) | **96 s**, 125.6 tok/s over the run | 228 s |
| Six requests decoding at once (1k-word prompts, 1,000 tokens each, prose at temperature 1.0, steady part) | **280.3–290.9 tok/s** with batched drafts, GPUs 90%; 303.0–303.9 with them off (`FNX_BATCH_DRAFTS=0`, below) | 198.7 tok/s |
| Code-edit prompt (`ref/edit1.bin`), decode (`serve/solo_decode.sh`: `tools/strata_stream.py`, greedy, 636 tokens, 6 runs) | **366.9–381.6 tok/s** (11.0 tokens a round, ~29.5 ms a round) | not measured |
| New-code prompt (`ref/code1.bin`), decode (same, 1,024 tokens) | 200.4–229.1 tok/s (3.8–4.6 tokens a round) | not measured |
| Prose reasoning at prod's sampling (`serve/solo_decode.sh`: `ref/prose2.bin`, temperature 0.6, 1,024 tokens, 5 runs) | 134.4–140.8 tok/s (2.1 tokens and ~15 ms a round; 110.0–113.4 on 2026-10-24) | not measured |
| A short pi session (`serve/pi_session.sh`: pi 1.0 fixes a bug and writes tests, 7–10 requests, 3k–9k-token prompts, sampled; three sessions) | 168.7–176.4 tok/s (3.2–3.4 tokens and ~19 ms a round) | not measured |
| One request, prose sampled at temperature 1.0 (1k-word prompt, 1,000 tokens) | 138.6–139.6 tok/s steady, GPUs 92–93% | 68.1 tok/s |
| Warm turn: +2k tokens on a 30k conversation, first token (`serve/warm_ttft.py`) | **0.95 s** median | 2.07 s |
| 30k random words, cold read (`serve/prompt_read.py`, 31,965 tokens) | **3,667 tok/s** (2,047 before the 2026-10-24/25 prompt-path work) | 2,454 tok/s |
| 114k-token prompt of source code, cold read (`serve/long_prompt.py`) | **3,373 tok/s** (33.8 s; 1,855 before), GPUs 93–97% | 2,971 tok/s (38.4 s), GPUs ~73% |
| 114k-token prompt, decode after it | 135.4 tok/s (150.1 on the reference's own tokens) | 72.9 tok/s |

Decode speed is tokens a round over the round's time: one verify window checks the draft layer's guesses (1 to 6, as
many as pay at the request's measured acceptance: a 3-row window ~13 ms, 5 rows ~16) or up to 15 from n-gram lookup
(~28 ms with 16), and costs the same however many survive. Code, and edits above all, draft well; prose drafts poorly
(47% of the drafts accepted at prod's sampling, 34% at temperature 1.0).

Several requests decoding at once share those windows, each with its own drafts (since 2026-10-26): a plan sizes every
lane's drafts by its own measured acceptance, the rows' cost at that lane count, and whether the window's layout has a
graph held (`src/batch_plan.hpp`). On the review-panel replay that is 7–9% more output than one row a lane (296 vs
318 s); on mixed traffic, an edit beside prose, it beats any fixed number of drafts a lane by 4–43%. Where it loses: six lanes of temperature-1.0 prose, whose drafts stand ~70% of the time at one a
lane, sit at the edge where the plan's cost model (~8% low on wide windows) picks drafts that do not quite pay: 5%
below one row a lane.

What the old pool could not do at all: six conversations of ~160k tokens each (960k positions of K/V) decoding at
once, 237.6 tok/s in all (`serve/arena_long_check.sh`), the K/V pool grown into 4.5 GiB of
card 2's expert memory; no conversation parked, paused or waiting.

sFNX is faster on every row now. Long cold prompts were Strata's until the prompt-path work of 2026-10-24/25 (attention
exchange in pieces and its merge deferred, the expert kernel on raw `mma.m8n8k4`, the recurrent core in chunks,
attention tiles of 32 positions, QSA in groups, K/V by thirds: 30k in the lab 3,176 -> 3,720 tok/s).

Upstream llama.cpp (a37cc46, same model, all layers on the three cards, `llama-bench`): 46.9 tok/s decode and 439
tok/s at pp512 with a layer split; 35.1 and 162 with its tensor split.

## Exactness

Against upstream llama.cpp (a37cc46, CUDA, greedy, its tokens), sFNX teacher-forced on the same tokens; the bar is
llama.cpp's own CPU backend forced the same way (`ref/fnx_logref`, `ref/img_logref`, `ref/lgrcmp.py`). sFNX as prod
runs it (int8 K/V, the prompt through the prompt path, windows of 5; `tools/lab.sh exact`, 2026-10-25, the same to
the byte on 2026-10-26's build, whose windows of 1, 2, 3, 11 and 16 rows also give the 5-row logits byte for byte):

| Prompt | | Top-1 agrees | KL mean | KL p99 |
| --- | --- | --- | --- | --- |
| Code, 300 steps | sFNX | 99.00% | 4.41e-4 | 7.1e-3 |
| | llama.cpp CPU | 98.33% | 4.85e-4 | 7.4e-3 |
| Code edit, 300 steps | sFNX | 100.00% | 3.91e-6 | 1.4e-4 |
| | llama.cpp CPU | 100.00% | 4.12e-6 | 1.9e-4 |
| The logo's source picture (`ref/img/sphinx.png`, a 601-token prompt), 200 steps | sFNX | 99.00% | 5.96e-3 | 1.57e-1 |
| | llama.cpp CPU | 98.50% | 6.38e-3 | 4.48e-2 |
| 62,794-token prompt (`ref/long64`), 200 steps | sFNX | 97.50% | 2.31e-3 | 2.64e-2 |

Windows of 11 and 16 rows give the 5-row windows' logits byte for byte, and every change of the 2026-10-25 decode
work kept them byte for byte. Long prompts are sensitive to the order sums are taken in (on `long64` reorders alone
moved KL between 1.8e-3 and 4.3e-3), so a change there is judged on the short prompts first: a precision loss in the
chunked recurrent core's decays (`__expf` under fast math) showed as code 7.0e-4 and `long64` 4.3e-3 until fixed.

- Temperature 0 through the sampler gives `fnx_run mtp`'s tokens exactly; sampled output passes chi-square tests
  against the model's distribution (`tools/sample_test.cpp`), alone and through batched windows with drafts
  (`SAMPLE_TEST_LANES=3`).
- Six lanes decoding together give each lane its tokens alone (`tools/batch_test.cpp`); a reply moved from the solo
  path into a batch slot gives the same text (`serve/promote_test.py`).
- A conversation restored from a checkpoint or from host memory continues with KL <= 3e-4 against reading it from
  scratch (`tools/cache_test.cpp`).

What is not exact: the sums between cards travel as fp16 pairs (fp32: `FNX_RING_F32=1`, KL 5.75e-4 -> 4.38e-4 on the
code prompt), and the kernels are built with fast math.

## K/V room: the arena

Six conversations of 262,144 tokens need 1.57M positions of K/V; the pool that fits beside every expert holds ~450k
(int8). Rather than park or pause conversations when they outgrow it, the served config (`FNX_ARENA=1` in
`serve/sfnx.json`) lets the pool borrow expert memory:

- The 43 layers whose experts share the largest size keep their VRAM slots (22,016 a card: 22.0 / 22.0 / 18.9 GiB) in
  one allocation with the K/V pool, right above the pool's own pages and ordered coldest first by a routing profile
  (`data/expert-routes.bin`, tallied over a code-review panel). The own pages are as many as the cards' free VRAM holds beside the slots and a
  reserve for the engine's other buffers (`FNX_ARENA_RESERVE_MIB`, 1,792, +600 for each 1,024 prompt rows past
  1,024: 3,592 at prod's 4,096). A card short of even the smallest own pool (another process holding VRAM on it)
  does not stop the engine: it puts that many slots' worth more of the coldest experts in RAM, places them again and
  says so (`fnx: VRAM short by N MiB ...`). Measured with 2 GiB held on card 0: 1,612 more experts in RAM, logits
  byte for byte the same.
- When a lane needs pages the free list lacks, the pool lends the next slots (at least 64 pages at a time, every card
  idle for it, 0.1-1.3 ms): their experts point back at their copies in page-locked RAM, and a window that picks one
  copies it into a slot first, as the Q8_0 build does (the expert cache). It can grow to ~2.9M positions.
- Slots come back once the top pages are free and no lend came in 30 s, refilled at once with the hottest experts
  that are out; while experts are lent, a lane idle for 120 s is parked to host memory to free its pages. With nothing
  lent every expert is in a slot and the windows skip the cache's kernels.


- Logits byte for byte against every expert in VRAM, on a 62k-token prompt and its decode (`tools/arena_check.sh`):
  identical with nothing lent, and with a 64-page own pool that lends during the read and the decode. Batched windows
  give each lane its tokens alone while lending (`bin/batch_test`, up to 718 slots lent on card 2).
- Pool pressure through the front end with a 461-page own pool: 9 of 9 conversations answer as alone; the pool lent 6
  times and gave back twice; no conversation parked, paused or waiting for a page.
- Six conversations of ~160k tokens each (960k positions) decoding together (`serve/arena_long_check.sh`): see the
  speed table. Card 2, which then held half the K/V, lent 5,201 slots (4.5 GiB of its experts), cards 0 and 1 2,464
  (not measured again with K/V by thirds).
- What it costs with nothing lent (`tools/arena_speed.sh`, against every expert in VRAM): windows the same (5-row
  verify 17.88 / 17.85 ms, six lanes 21.11 / 21.09 ms), one request on new code the same (187.1 / 186.3 tok/s), on a
  code edit 4.1% slower (299.8 / 312.5): the 72 GiB of page-locked expert copies, which cost a code edit 4.3% with
  the cache off too and batched windows nothing.
- What lent experts cost a batch (six at once, coldest experts in RAM): -3.9% at 8 GiB, -5.1% at 16 GiB.

Prompts are read 4,096 rows a chunk (`FNX_PF_ROWS`, 1,024 to 8,192): ~20% faster than 1,024 on anything longer than
1k tokens (2,486 against 2,065 tok/s on 8k, `tools/chunk_probe.sh`; 8,192 rows 2,539), for ~1.1 GiB more a card. To
fit that at 262k, prod starts with the coldest 3 GiB of experts in RAM (`FNX_EXPERT_RAM_GIB=3`, ~1 GiB a card), so the
cache's kernels run in every window (about 2% of decode). On Q8_0, whose experts already sit behind the cache, the
larger chunk costs nothing extra. Logits at 4,096 rows against llama.cpp on the 62k prompt: KL mean 3.97e-3, top-1 97%
(1,024 rows: 3.10e-3, 98%; the bar, llama.cpp's own q8_0 K/V: 4.98e-3).

## Q8_0

The same model in Q8_0 (six files, 176 GiB on disk) does not fit in VRAM,
so part of its routed experts stay in RAM. It is not the served config (`sfnx.service` runs `serve/sfnx.json`, the
Q4_K_XL model); `serve/run.sh serve/sfnx-q8.json` runs it on port 8095. Settings, all in the engine's environment:

- `FNX_EXPERT_RAM_GIB=N` with `FNX_EXPERT_PROFILE=FILE`: the coldest N GiB of experts by the profile (routing tallies
  written by `FNX_ROUTE_COUNT=1`, `FNX_ROUTE_FILE`; decode and prompt picks weighted equally, per byte) are kept in
  page-locked RAM. Every expert has a copy there; VRAM slots per expert size hold the ones in use, and a decode window's
  experts not in a slot are copied in before the MoE runs. On Q4_K_XL, output was identical to all experts in VRAM
  (KL 0), and decode on `ref/code1.bin` went from 176 to 160.6 / 155.5 / 146.7 tok/s with 8 / 16 / 24 GiB in RAM.
- `FNX_KV8=1`: K/V rows as int8 (q8_0 groups; pooled QSA keys stay f16): 449,856 positions in the f16 pool's bytes.
  KL against llama.cpp CUDA with f16 K/V, teacher-forced: code1 7.1e-4 (f16 K/V 6.7e-4, llama.cpp q8_0 K/V 5.1e-4), edit1
  3.7e-6 (5.1e-6, 2.3e-6), long64 (62k tokens) 3.10e-3 (4.16e-3, 4.98e-3). Five lanes at once: the same speed with
  either (229 against 230 and 197 against 199 tok/s).
- The prompt path with experts in RAM: `FNX_PF_PASS` (default 4, at most 8) chunks of 1,024 rows go through each layer
  before the next, and the layer's experts not in a slot are copied into a VRAM stage (two, `FNX_PF_STAGE_MIB` each,
  default 384) while the layer before computes; reads under `FNX_PF_STAGE_ROWS` tokens (default 2048) take the experts
  in place. Q8_0, 52 GiB in RAM, int8 K/V, a 62,793-token prompt (`fnx_run`): 496 -> 1,252 tok/s, logits identical.

`serve/sfnx-q8.json`: 58 GiB of experts in RAM (52 until windows of up to 16 rows took ~300 MiB more a card for the
recurrent layers' snapshots: at 52 and 55 the first long prompt's buffers no longer fit), int8 K/V, a 64 GiB
conversation cache. Lab decode at 58 GiB, lookup drafts up to 15 (`tools/lab.sh decode`, greedy, 2026-10-25): code
edit 137.8 tok/s (130.4 with lookup capped at 7), new code 111.5. The table below was measured at 52 GiB and lookup
7, against Strata on the same files with prod's settings:

| Workload | sFNX | Strata |
| --- | --- | --- |
| One request (1k-word prompt, 1,000 tokens; before the prompt passes above) | 88.9 tok/s | 53.1 tok/s |
| Five at once, same (steady / over the run / first token median) | 180.3 / 123.8 tok/s / 7.7 s | 143.2 / 79.4 tok/s / 20.1 s |
| New code (`ref/code1.bin`) through the API, temperature 0, three runs | 116.4 / 122.3 / 118.7 tok/s | 59.7 / 67.5 / 69.3 tok/s |
| A code edit (`ref/edit1.bin`), same | 139.9 / 146.2 / 144.4 tok/s | 63.6 / 76.9 / 79.5 tok/s |

Both start in ~310 s.

Exactness on Q8_0 (`tools/q8_exact.sh`): llama.cpp's CPU backend (a37cc46; the Q8 model does not fit the
cards for llama.cpp) generates 300 greedy steps on the Q8_0 GGUF, and sFNX's Q8 path as `serve/sfnx-q8.json` runs it
(52 GiB of experts in RAM, int8 K/V, the prompt through the prompt path, windows of 5) is teacher-forced on its tokens:

| Prompt | Top-1 agrees | KL mean | KL p99 |
| --- | --- | --- | --- |
| Code, 300 steps | 99.00% | 5.21e-4 | 6.45e-3 |
| Code edit, 300 steps | 100.00% | 2.79e-6 | 1.06e-4 |

At today's build (2026-10-25, 49 GiB in RAM, the lab's `logits` against the same CPU tokens): code 99.00% / 5.34e-4
/ 8.3e-3, code edit 100.00% / 2.79e-6 / 2.1e-5.

Within the Q4 bar above (llama.cpp's CPU backend against its CUDA one: 4.85e-4 and 4.12e-6), as sFNX's Q4 path is.

## The lab

For engine work the model stays loaded and each change costs seconds (`tools/lab.sh`):

- **The holder** (`bin/fnx_hold`) loads the model once: the dense weights on the cards and every routed expert's copy
  in host memory near its card, all from the CUDA virtual memory API at fixed addresses (`src/store.*`). It runs no
  kernel. 46 s from the page cache; 7.5 GiB on the cards, 74 GiB of host copies (Q4).
- **A worker** (`bin/fnx_lab --attach`) maps that memory at the same addresses, takes the cards' tables over a socket,
  places the experts by its own settings, and runs commands from `build/libfnx_lab.so`: ~7 s to attach. A worker
  whose kernels fault ends, and the next command attaches a new one; the model is not loaded again.
- **Settings:** prod's are the defaults. `cmd "set VAR=..."` reloads the engine library (engine settings, at once);
  `FNX_ARENA`, `FNX_EXPERT_RAM_GIB` and `FNX_EXPERT_PROFILE` place the experts again in place (~3 s). A rebuilt
  library: `cmd reload`. A changed loader: `make lab`, `unload`, `start`.
- **Standard commands:** `read N...` (a prompt read's time, the GPUs sampled); `exact [NAME]` (logits on the 62k
  prompt, code1, edit1 and the picture through the prompt path against the held model's llama.cpp references and
  a base, ~30 s; `exact base` writes the base); `decode [N] [K] [LK]` (one request alone as prod decodes it, edit1 and
  code1: tok/s and a round's verify / draft / commit time); `batch [GEN]` (batched windows give each lane its tokens
  alone, 1 to 6 lanes); `compare A.lgr B.lgr`; and every `fnx_run` mode (`rows`, `bench_step`, `mtp`, `spec`,
  `check`, `logits`, ...). A command's own setup: ~2 s for the engine (the arena filled from host memory by one kernel
  per card), ~1.2 s for its graphs.
- **Through prod:** `serve/solo_decode.sh` (edit1 and code1, three runs each, with the engine's own line per request)
  and `serve/pi_session.sh` (a short real pi session on `bench/pi-task`).
- **Debugging:** each command has a watchdog (`FNX_LAB_TIMEOUT`, default 600 s); `FNX_LAB_GDB=1` runs the worker under
  gdb. compute-sanitizer and `CUDA_LAUNCH_BLOCKING=1` cannot run the engine: they run one kernel at a time, and the
  ring sums' kernels wait on the other cards'. A fault on the cards: the `check` mode launches without graphs.

## Tests

| What | How |
| --- | --- |
| API surface through the front end (10 checks) | `serve/api_smoke.py` |
| Pictures through the front end (6 checks) | `serve/image_test.py` |
| K/V pool under pressure without the arena (32k context, 6 x 9k conversations, a decoding slot paused and resumed), and cancel | `serve/cancel_pool_check.sh` |
| The arena: logits against every expert in VRAM, nothing lent and lending; batched = alone while lending | `tools/arena_check.sh` |
| Six ~160k-token conversations decoding at once | `serve/arena_long_check.sh` |
| Draft layer after batched windows in freshly lent pages (the NaN it used to meet) | `serve/gap_check.sh serve/sfnx-gap.json "plain="` |
| Batched windows = alone, with and without drafts (every token up to each reference's end) | `bin/batch_test`, `tools/lab.sh batch` |
| Batched drafts' sampler against plain sampling, in distribution (3 lanes, open1, 600 seeds: every p >= 0.34) | `SAMPLE_TEST_LANES=3 bin/sample_test ...` |
| The batched-draft row plan against a brute force over every split | `bin/batch_plan_test` |
| The conversation cache where prompts share much: two conversations alternating on a lane, a system prompt changed near its end | `tools/lab.sh cmd cache2` |
| The front end's own tests | `.venv/bin/python -m pytest -q front/` (970 passed, 19 skipped, ~6 min; they run the front end against fake engines and do not start `bin/sfnx`) |

## Known limits

- One model, one box: three cards, sm_70; UD-Q4_K_XL served, Q8_0 with part of its experts in RAM.
- Where sFNX's API differs from Strata's: `logit_bias` is refused (the engine does not offer it, so the front end
  refuses the request); disk sessions (`POST /slots/0?action=save|restore`, the engine's SAVE/RESTORE) and
  `POST /v1/vram` (the engine's VRAM) are not supported: the front end refuses slot files with batch slots on, and
  the engine answers `ERR ... not supported by sfnx` to SAVE, RESTORE and VRAM. A request's `strata_prefix` (a
  pinned checkpoint at a shared prefix) is accepted and does nothing; the engine logs that once. Strata's per-request
  engine tuning (`strata_tune`) and `experimental_speed_projection` are ignored.
- The arena's limits: a prompt read with experts lent reads those in place from RAM (the prompt path's
  stages are sized from the experts out of a slot at load, none with the arena), not measured yet; the page-locked
  expert copies cost a solo code edit ~4% even with nothing lent; and requests read one at a time without decoding (one
  token each) all land on the same lane, so each parks the one before (~1 s each).
- Batched windows carry each lane's drafts (since 2026-10-26), sized by a plan that knows each lane's acceptance, the
  row and draft costs and which window layouts have a graph held; it lands within 3% of the best fixed count of drafts
  a lane on every content measured at 2-5 lanes, and on mixed traffic well ahead of any fixed
  count. Where it loses: six lanes of temperature-1.0 prose, ~4% under one row a lane.
  `FNX_BATCH_DRAFTS=0` gives the old one row a lane, whose draft layer then never sees batched tokens.
- Once (a reasoning-close retry) the engine failed with a message that was lost; failures
  now print and exit at once (the front end restarts the engine), so the next one will say what it was.
- VRAM is nearly all used: after start (every lane, every window graph and the prompt path's buffers made before the
  first request, so it stays flat; with batched drafts a verify graph for every row total) cards 0 and 1 have 284 MiB
  free, card 2 2,540 MiB beside the picture encoder; K/V beyond the own pool comes from expert slots, not from free
  VRAM.
  Parked conversations live in pageable memory: page-locked memory costs 2 MiB of every card's VRAM per GiB.
- A prompt read from scratch on a lane that held another conversation can differ in the last bits from the same read
  on a fresh engine (first-token KL ~1e-4; greedy text can part at a near-tie): something of the lane's earlier use
  reaches the arithmetic, not traced yet (a guess: the prompt path's padding rows). Two fresh engines agree exactly.
