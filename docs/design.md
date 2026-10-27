# Design

How the model sits on three cards, and what each part of a request costs. Measured on the reference box — three
V100-32GB cards on PCIe Gen3, no NVLink — unless a line says otherwise.

## Splitting every layer

Strata and llama.cpp's layer split give each card a third of the layers. For one request the cards take turns; with
several requests in flight each card can hold a different one (Strata's `--batch-groups 3`). sFNX gives every card a
third of **every** layer:

- **Experts.** Each of the 512 experts per layer has a third of its FF width on each card. All 24,576 experts are in
  VRAM until the conversations need room for K/V ([docs/memory.md](memory.md)).
- **Recurrent (Gated DeltaNet) layers.** 16 value heads per card.
- **Attention.** The K/V cache is split by position: blocks of 4 positions dealt to cards 0, 1, 2 in turn, a third
  each. Every card attends over its own positions and the partial results merge. Thirds rather than halves for card 2
  read 30k tokens 2.5% faster.
- **Output head.** A third of the vocabulary per card.
- **Copied on every card.** The hyper-connection mixes, the router, the embeddings.

That is three cards. The split follows one rule for any count the build allows (`src/card_split.hpp`; 2, 3, 4, 6 or 8,
[docs/config.md](config.md#card-count)): the value heads in key-head order dealt in equal runs, a key head on two cards
when a run starts inside its value heads; the FF width in 32-value blocks, the first cards one block more when the
count does not divide 20; the query heads, K/V blocks and vocabulary 1/n each. Only two and three cards have run.

Between the cards: two sums per layer, 96 per window, passed around a ring as fp16 pairs (three cards; the other
counts' decode sends fp32). On n cards a sum takes n - 1 hops to reduce and n - 1 to spread. A value past fp16's range
is an error, never a silent overflow. `FNX_RING_F32=1` sends them as fp32 on three cards too (KL 5.75e-4 → 4.38e-4 on
the code prompt).

Decode moves the sums with the GPU's own threads; prompts move them with the copy engines. The ring kernels take ~4.5 ms
of a 22.5 ms six-row window on card 0, waiting for the slowest card included.

## Why not NCCL, and why not tensor parallel

Serving engines that split tensors (vLLM, SGLang, TensorRT-LLM) assume NVLink, and split attention by heads, which
needs a head count the card count divides — 2, 4 or 8 cards, not 3. Over PCIe, NCCL costs ~85–110 µs per small
all-reduce on this box: ~10 ms a token in sums alone, for 96 of them.

So the sums are GPU threads storing into the next card's memory inside the window's CUDA graph. No kernel boundary, no
collective, no library in the way. The price is that the design is specific to this model and this box.

## Decode

One verify window per round checks up to 16 rows. Drafts come from two places:

- **The model's MTP draft layer** drafts a chain of `--spec` tokens (prod 6). Its routed experts are Q4_0 made from the
  BF16 checkpoint (`tools/mtp_experts.py`); Strata's Q2_0 build (`FNX_MTP_EXPERTS=q2_0`) gets 59% of new code's drafts
  accepted against 63%.
- **N-gram lookup** in the context drafts as many tokens as the match is long, 3 to 15. It pays on text that repeats:
  code edits.

A window costs about the same however many drafts survive — ~13 ms at 3 rows, ~16 at 5, ~28 with 16 — so acceptance
sets the speed. Code and edits draft well; prose drafts poorly: 47% of drafts accepted at prod's sampling (temperature
0.6), 34% at temperature 1.0.

Temperature, top-p, top-k, min-p and the repeat, frequency and presence penalties are applied as Strata does. Sampled
drafts are accepted by rejection sampling, so the output distribution is the model's
([docs/verification.md](verification.md)).

## Six conversations at once

K/V lives as int8 in a paged pool, 192 positions a page, shared by six lanes. The decoding lanes share each window,
each lane with its own drafts; a prompt being read lets the others decode between its chunks. The pool grows into the
coldest experts' VRAM when the conversations need it, so all six can hold their full 262,144-token context at once
([docs/memory.md](memory.md)).

**Batched drafts.** A plan sizes every lane's drafts by that lane's measured acceptance, the rows' cost at that lane
count, and whether the window's layout has a graph held (`src/batch_plan.hpp`). On a review-panel replay it gives 7–9%
more output than one row a lane; on mixed traffic — an edit beside prose — it beats any fixed count of drafts a lane by
4–43%, and lands within 3% of the best fixed count at 2–5 lanes. Where it loses: six lanes of temperature-1.0 prose,
whose drafts stand ~70% of the time at one a lane, sit at the edge where the plan's cost model (~8% low on wide
windows) picks drafts that do not quite pay — 5% under one row a lane. `FNX_BATCH_DRAFTS=0` gives one row a lane, whose
draft layer then never sees batched tokens.

## The prompt path

Prompts are read in chunks of 4,096 rows (`FNX_PF_ROWS`, a multiple of 16 up to 8,192): ~20% faster than 1,024 on
anything longer than 1k tokens — 2,486 against 2,065 tok/s on 8k — for ~1.1 GiB more per card. 8,192 rows measures
2,539.

Experts run on the tensor cores (f16 dequantization plus `mma.m8n8k4`); dense matrices are f16 cuBLAS GEMMs. The rest
of the speed on long prompts comes from doing the attention exchange in pieces and deferring its merge, keeping the
recurrent core in chunks, attention tiles of 32 positions, and QSA in groups.

To fit 4,096 rows at 262k, prod starts with the coldest 3 GiB of experts in RAM (`FNX_EXPERT_RAM_GIB=3`, ~1 GiB a
card), so the expert cache's kernels run in every window — about 2% of decode. On Q8_0, whose experts already sit
behind the cache, the larger chunk costs nothing extra.

## The conversation cache

Checkpoints at message boundaries and before late notes (`keep=N`), and every 4,096 tokens inside a long first message.
A system prompt changed near its end — an edited `AGENTS.md` — is read again only from the last checkpoint before the
change. Conversations that leave a lane, or would lose 2,048 tokens or more to the next prompt on it, are parked in
host memory (128 GiB), least recently used out first. Parked conversations live in pageable memory: page-locked memory
costs 2 MiB of every card's VRAM per GiB.

## Pictures

`bin/sfnx-vision` (llama.cpp's mtmd and the model's mmproj) encodes them on GPU2, or the CPU if that card cannot take
it. A picture's cells get the encoder's rows and llama.cpp's interleaved M-RoPE positions. The cache matches a picture
by its hash.

## Quants

The engine is built per model, not per quantization: it reads the GGUF's tensor types and dispatches on them. Two sets
matter.

- **Routed experts:** Q4_0, Q4_K, Q5_1, Q5_K or Q8_0 decode; the prompt path's tensor-core kernel takes Q4_K, Q5_K,
  Q5_1 or Q8_0. The kernels are templated per type — anything else throws
  (`moe: expert type N not handled`) rather than dequantizing generically.
- **Dense matrices:** Q4_K, Q5_K, Q5_1, Q6_K or Q8_0 (`mmvq: weight type N not handled`).

Some tensors are read by a kernel written for one layout, so a GGUF must keep them at Q8_0: **token embeddings**, the
**shared expert**, and the **draft layer's projections**. Splitting an expert by FF width also needs 32-value blocks.
Unsloth's UD-Q4_K_XL and the Q8_0 build both satisfy this; those two are what runs and what the checks hold
([docs/verification.md](verification.md)). A quant that does not is a new kernel, not a config change — and it fails
at load, naming the tensor.

## Hardware

The cards are used datacenter V100s: `sm_70`, no CUDA 13, no FP8, capped at 200 W each (110–150 W each sampled in the
speed runs). That is why an engine like this had to be written for them, and it is what makes ~$3,000 of cards run a
model this size out of VRAM.
