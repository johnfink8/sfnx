# Verification

What "correct" means here, how it is measured, and every check that guards it.

## The bar

sFNX is compared against upstream llama.cpp (a37cc46, CUDA, greedy, on llama.cpp's own tokens) with sFNX
teacher-forced on the same tokens. **The bar is not zero: it is llama.cpp's own CPU backend forced the same way.** If
sFNX sits inside the CPU backend's distance from the CUDA one, it is within the tolerance llama.cpp accepts for itself.

Passing is slack, not equality: KL mean at most `kl_slack` × the bar's, and top-1 at most `TOP1_SLACK` = 1.5 points
under it — both held in `tools/fnx_run.cpp` beside the bars themselves. `kl_slack` is 1.25 for code1, edit1 and the
picture: sFNX's fp16 ring sums put the Q8_0 path at 1.10×, and the one precision loss caught so far (fast-math decays
in the recurrent core) reached 1.44×. For `long64` it is 1.0, because summation order alone moves its KL between 1.8e-3
and 4.3e-3 and there is no room left to give.

Tools: `ref/fnx_logref` and `ref/img_logref` produce the references, `ref/lgrcmp.py` compares, `tools/lab.sh exact`
runs the whole set (~30 s). Q4 as prod runs it: int8 K/V, the prompt through the prompt path, windows of 5 rows.

| Prompt | | Top-1 agrees | KL mean | KL p99 |
| --- | --- | --- | --- | --- |
| Code, 300 steps | sFNX | 99.00% | 4.41e-4 | 7.1e-3 |
| | llama.cpp CPU | 98.33% | 4.85e-4 | 7.4e-3 |
| Code edit, 300 steps | sFNX | 100.00% | 3.91e-6 | 1.4e-4 |
| | llama.cpp CPU | 100.00% | 4.12e-6 | 1.9e-4 |
| The logo's source picture (`ref/img/sphinx.png`, a 601-token prompt), 200 steps | sFNX | 99.00% | 5.96e-3 | 1.57e-1 |
| | llama.cpp CPU | 98.50% | 6.38e-3 | 4.48e-2 |
| 62,794-token prompt (`ref/long64`), 200 steps | sFNX | 97.50% | 2.31e-3 | 2.64e-2 |

Windows of 1, 2, 3, 11 and 16 rows give the 5-row windows' logits byte for byte.

**Long prompts are sensitive to the order sums are taken in.** On `long64` reorders alone moved KL between 1.8e-3 and
4.3e-3, so a change there is judged on the short prompts first. That is how a precision loss in the chunked recurrent
core's decays (`__expf` under fast math) was caught: it showed as code 7.0e-4 and `long64` 4.3e-3 until fixed.

Logits against llama.cpp on the 62k prompt by prompt-chunk size: 4,096 rows KL mean 3.97e-3, top-1 97%; 1,024 rows
3.10e-3, 98%; the bar — llama.cpp's own q8_0 K/V — 4.98e-3.

## Other guarantees

- Temperature 0 through the sampler gives `fnx_run mtp`'s tokens exactly.
- Sampled output passes chi-square tests against the model's distribution (`tools/sample_test.cpp`), alone and through
  batched windows with drafts (`SAMPLE_TEST_LANES=3`).
- Six lanes decoding together give each lane its tokens alone (`tools/batch_test.cpp`); a reply moved from the solo
  path into a batch slot gives the same text (`serve/promote_test.py`).
- A conversation restored from a checkpoint or from host memory continues with KL ≤ 3e-4 against reading it from
  scratch (`tools/cache_test.cpp`).

## What is not exact

- The sums between cards travel as fp16 pairs. `FNX_RING_F32=1` sends decode's as fp32: KL 5.75e-4 → 4.38e-4 on
  the code prompt on three cards. On two cards fp16 leaves code and edit slightly over the bar (code KL 7.5e-4 against 6.1e-4, edit 7.1e-6 against 5.2e-6; the same top-1 as fp32); fp32 brings them within
  it, at ~30% of one request's decode. The prompt path's sums stay fp16 on every count (its copy-engine ring has no fp32
  form); the 62k prompt, read through it, is within the bar on two and three. On n cards a part is rounded to f16 at
  each of n - 1 hops, so on four or more it is the first suspect if `long64` fails.
- The kernels are built with fast math.
- A prompt read from scratch on a lane that held another conversation can differ in its last bits from the same read on
  a fresh engine: first-token KL ~1e-4, and greedy text can part at a near-tie. Something of the lane's earlier use
  reaches the arithmetic; not traced. Two fresh engines agree exactly.

## Q8_0

`tools/q8_exact.sh`. The Q8 model does not fit the cards for llama.cpp, so llama.cpp's CPU backend generates 300 greedy
steps on the Q8_0 GGUF and sFNX's Q8 path — as `serve/sfnx-q8.json` runs it, 52 GiB of experts in RAM, int8 K/V, the
prompt through the prompt path, windows of 5 — is teacher-forced on its tokens.

| Prompt | Top-1 agrees | KL mean | KL p99 |
| --- | --- | --- | --- |
| Code, 300 steps | 99.00% | 5.21e-4 | 6.45e-3 |
| Code edit, 300 steps | 100.00% | 2.79e-6 | 1.06e-4 |

On the current build at 49 GiB in RAM (the lab's `logits` against the same CPU tokens): code 99.00% / 5.34e-4 / 8.3e-3,
code edit 100.00% / 2.79e-6 / 2.1e-5. Both inside the Q4 bar above — llama.cpp's CPU backend against its CUDA one,
4.85e-4 and 4.12e-6 — as sFNX's Q4 path is.

## The 27B

Qwen3.8-27B Q8_0 (`make FNX_MODEL=27b`, [docs/config.md](config.md#model)). Its bar is its own: the widest gap
between llama.cpp's (a37cc46) own ways of running the 27B, each teacher-forced on its CUDA greedy run's tokens.

- **Its CPU backend against CUDA** (`ref/fnx_logref` with `FNX_LOGREF_GPU=0`; the picture under llama.cpp's own
  encoder, `ref/img_logref` with `IMG_LOGREF_GPU=0`), as Flash-Next's bar.
- **Its CUDA with every product through cuBLAS** against its default CUDA. On a V100, llama.cpp's default multiplies
  a decode row's Q8_0 matrices with int8 products of activations quantized to q8_1 (MMVQ, and MMQ under 64 rows),
  and its CPU backend quantizes them to q8_0 the same way; a prompt's batches already go through cuBLAS in f16.
  `ref/llama_cublas_only.patch` turns the quantized products off under `FNX_CUBLAS_ONLY=1` (and asserts none ran).
  sFNX multiplies f16 activations, as this path does.
- **For the 62k prompt**, where the CPU backend would take hours: its q8_0 K/V against its f16 K/V, as Flash-Next's
  long bar.

The dense model's two llama.cpp backends agree far more closely than Flash-Next's do: code KL 8.95e-5 against
Flash-Next's 4.85e-4, edit 1.48e-7 against 4.12e-6. Its cuBLAS path is as far from its default as sFNX is (code
1.25e-4), so a bar from the CPU backend alone (8.95e-5) would be one draw from a spread llama.cpp
itself does not stay inside.

`bin/fnx_run MODEL exact`, as `serve/sfnx-27b.json` runs it: f16 K/V, the prompt through the prompt path (f16 scores),
windows of 5. Measured 2026-10-29:

All against llama.cpp's default CUDA run:

| Prompt | | Top-1 agrees | KL mean |
| --- | --- | --- | --- |
| Code, 300 steps | sFNX | 100.00% | 1.25e-4 |
| | llama.cpp CPU | 100.00% | 8.95e-5 |
| | llama.cpp cuBLAS | 100.00% | 1.25e-4 |
| Code edit, 300 steps | sFNX | 100.00% | 1.44e-7 |
| | llama.cpp CPU | 100.00% | 1.48e-7 |
| | llama.cpp cuBLAS | 100.00% | 1.23e-7 |
| The logo's source picture, 200 steps | sFNX | 99.50% | 7.56e-4 |
| | llama.cpp CPU | 99.50% | 4.69e-4 |
| | llama.cpp cuBLAS | 99.00% | 8.91e-4 |
| 62,794-token prompt (`ref/long64`), 200 steps | sFNX | 100.00% | 2.65e-4 |
| | llama.cpp q8_0 K/V | 99.50% | 2.08e-4 |
| | llama.cpp cuBLAS | 100.00% | 1.04e-4 |

sFNX sits as far from llama.cpp's cuBLAS path as from its default (code 1.16e-4, edit 1.28e-7, picture 9.16e-4) and
from its CPU backend (code 1.13e-4): about the distance between any two of llama.cpp's own.

Windows of 1, 2, 3, 11 and 16 rows give the 5-row windows' logits byte for byte.

On two cards (`FNX_NCARD=2`, GPUs 1 and 2, its own lab directory) the same: code 100% / 1.34e-4, edit 100% / 1.46e-7,
the picture 99.0% / 5.95e-4, all within the bar; long64 100% / 3.06e-4, over it as on three; the windows byte for
byte. Its batched windows and its sums (`bin/s27_ring_test`) pass there too. Four cards build and have never run.

The bar takes the CPU backend's top-1 (the highest of the three) and the widest KL, with Flash-Next's slack on top
(1.25x; 1.0 for long64).

**Code, edit and the picture are within the bar; long64 is not, and the exact check says so** (`FAIL`, so no base is
written). Every prompt's top-1 agreement is the bar's or better. long64's KL is 1.27x the bar, where the slack is 1.0,
and 2.5x the distance between llama.cpp's own two CUDA paths there (1.04e-4): over 62k tokens sFNX drifts further
than either. Not yet found; the suspects are the f16 scores over 62k keys, the order of the chunked recurrent sums and
the fp16 sums between the cards.

How much these move with each choice:

| Setting | Code | Picture | long64 |
| --- | --- | --- | --- |
| The default, f16 scores | 1.25e-4 | 7.56e-4 | 2.65e-4 |
| `FNX_PF_SCORES=f32` | 1.13e-4 | 8.82e-4 | 2.89e-4 |
| 1,024-row prompt chunks (`FNX_PF_ROWS=1024`) | | | 2.88e-4 |
| The decode decay through `__expf` (before this build) | 9.48e-5 | | |

Each choice moves the numbers by about as much as long64's gap, and none brings it in.

The fp16 sums between the cards are not it: `FNX_RING_F32=1` moved edit1 by 1.45e-7 and left its distance from
llama.cpp where it was.

**What the checks found on the way:**

- **Every prompt pass after the first re-read the prompt's start.** long64 took the end-of-text token from its first
  step: top-1 21.5%, KL 3.67. Fixed: the position-to-token mapping takes the history's length once before the passes,
  as `prefill.cu` does. Prompts of 8,192 tokens or fewer were never affected.
- **Decode's recurrent decay went through `__expf`.** A prompt read through windows drifted from llama.cpp: edit1 KL
  1.1e-6, against 2.25e-7 with the decay through double, as `gdn_prompt` already took it. This is the 27B's only; its
  build has it behind `#if FNX_27B`.

`bin/fnx_run MODEL exact` runs the set, as `tools/lab.sh exact` does for Flash-Next (the lab holds Flash-Next only).
The 27B's references are `ref/lgr/llama_cuda_Qwen3.8-27B-Q8_0_<prompt>.lgr` and `ref/img/sphinx_Qwen3.8-27B-Q8_0.*`,
made locally like the rest of `ref/lgr/`; the bar's, `llama_cpu_*` and `llama_cublas_*` (llama.cpp built with
`ref/llama_cublas_only.patch`, run with `FNX_CUBLAS_ONLY=1`) beside them. `tools/s27_cmp.py` compares a layer dump
(`bin/fnx_run MODEL check`) with `ref/fnx_ref`'s: on a 19-token prompt every layer's mixer and FFN sums are at cosine
0.99957 or closer.

Checked on the 27B as on Flash-Next:

- **Batched windows** give each lane its tokens alone, plain and with drafts (`bin/batch_test`, 1 / 2 / 3 / 6 lanes).
  Each batched row is rotated by its own lane's M-RoPE table.
- **The speculative sampler** matches plain sampling through three batched lanes with drafts (`SAMPLE_TEST_LANES=3
  bin/sample_test`, open1, 600 seeds: every p >= 0.116).
- **Its sums** (`bin/s27_ring_test`): the fp32 ring gives the cards' sum exactly and the fp16 one (three a word)
  within fp16's rounding of it, at 1, 2, 5 and 16 rows, with no error bit in the timed sums or the checked ones.
- **The API and pictures through the front end:** `serve/api_smoke.py` 10 of 10, `serve/image_test.py` 6 of 6
  (three requests at once among them), and `pytest front/` 595 passed.

**Flash-Next on the shared engine gives main's logits.** With what both models share moved into
`engine_common.cu` (2026-10-29), Flash-Next's exact check against a base written by main's build (ada6d6f) is byte
for byte identical on all four prompts, and every window size gives the 5-row windows' logits. Its batched windows,
its sampler and the top-k ties pass. Its decode is main's: edit1 366.1 against 365.5 tok/s, code1 180.4 against
177.7, prose2 with prod's sampling 120.9 against 121.6 (GPUs 87% busy, 115 W). The 27B's logits on the shared engine
are byte for byte the ones before the move.

## Other card counts

Four, six and eight cards build (`make FNX_NCARD=n`, [docs/config.md](config.md#card-count)) but **have never run**:
nobody here has the cards. Checked without them, on the CPU:

- the split rule (`src/card_split.hpp`, static checks in every build): on every supported count each value head is on
  one card beside its key head, the key heads and FF blocks are covered in order, no card is empty; for two and three
  cards the rule gives exactly the splits the bases were made with;
- the prompt path's ring (`bin/pf_ring_plan_test`): its steps for two and three cards are the sequences the
  hand-written code ran, and on every supported count the cards' steps run against each other complete eight sums in
  a row over the two buffer sets and the sequence-numbered flags, in 500 random interleavings: every part holding
  every card's addend once, nothing read before the flag that covers it for its sum, no slot written over before the
  sum before had read it, without a deadlock.

Not checked anywhere: the decode ring and the attention merge on more than three cards. They are the same loops on
every count (n - 1 hops to reduce, n - 1 to spread), run so far with n = 2 and 3. Nor speed: each sum takes 2n - 2
hops, so more cards add latency to each of a window's 96 sums while each card's share of the weights shrinks.

To validate count n (n cards, `iommu=pt` or the IOMMU off, and peer access between every pair of cards:
[docs/config.md](config.md#boot-requirement)), with nothing else on the cards:

1. Build: `FNX_NCARD=n make -j lab bin/sfnx test-bin ref`. The CPU checks: `bin/pf_ring_plan_test`,
   `bin/batch_plan_test`, `bin/sfnx_input_test`, `bin/card_check_test`, `tools/build_guards_check.sh`.
2. llama.cpp's references, before the lab takes the cards: `ref/fnx_logref MODEL ref/P.bin STEPS ref/lgr/llama_cuda_P.lgr`
   with MODEL the UD-Q4_K_XL GGUF's first shard, for code1 and edit1 (300 steps) and long64 (200). `exact` prints the
   command for any it misses; the picture's reference is in git.
3. A lab directory of the count's own (its bases differ from three cards'), on every `lab.sh` call:
   `export FNX_LAB_DIR=/tmp/fnx-lab-$UID-${n}card`, then `tools/lab.sh start`.
4. `tools/lab.sh exact base`: logits against llama.cpp on long64, code1, edit1 and the picture. It writes a base only
   when every prompt is within the bar: KL mean at most 1.25x llama.cpp's own CPU-against-CUDA KL for code1, edit1 and
   the picture (1.0x for long64), top-1 at most 1.5 points under it.
5. `tools/lab.sh exact` (against that base, and windows of 1, 2, 3, 11 and 16 rows byte for byte),
   `tools/lab.sh batch` (batched windows give each lane its tokens alone) and `tools/lab.sh decode` (speed, with the
   GPUs' use).
6. If step 4 fails on KL alone: fp32 sums in a directory of their own, to tell the ring's rounding from a real fault.
   `tools/lab.sh unload`, then with `FNX_LAB_DIR=/tmp/fnx-lab-$UID-${n}card-fp32`:
   `tools/lab.sh start "" "-E FNX_RING_F32=1"` and `exact base`. Two cards need this to pass step 4.
7. Through the front end, `tools/lab.sh unload` first (four cards: `serve/sfnx-4card.json`; another count: a copy with
   its `"gpu"` list and vision card): `serve/run.sh serve/sfnx-4card.json`, then `serve/api_smoke.py
   http://127.0.0.1:8095` and `serve/image_test.py http://127.0.0.1:8095/v1`.

A failure in step 4 with the rest clean points at the arithmetic (the sums' precision, a split); a hang, a `ring` or
`attn` error, or a timeout in the log at the transport (peer access, the IOMMU, a buffer sized for fewer cards).
One place rests on timing rather than flags: with `FNX_PF_OVERLAP=1` (the default) the attention exchange takes a
sequence number between two ring sums, so two attention layers' sums can use the same buffer set back to back
(`bin/pf_ring_plan_test` does not model it). A hang or a wrong long64 that goes away with `FNX_PF_OVERLAP=0` points
there. Six cards put four key heads on two cards each, and eight give each card the smallest slices (FF 64 or 96
wide); a count that fails where four passes points at those. Above three cards the engine says at start that the
count has not run on hardware yet.

## Rope scaling

`FNX_ROPE_SCALE=S` ([docs/config.md](config.md)) is llama.cpp's YaRN, so a scaled build is compared with llama.cpp
run with the same scaling: `ref/fnx_logref` and `ref/img_logref` read `FNX_ROPE_SCALE` and `FNX_YARN_ORIG_CTX` by the
engine's rules, and with them set `exact` compares against `ref/lgr/llama_yarn_S[_origN][_MODEL]_PROMPT.lgr` (its
hint for a missing one carries the setting). A scaled build's lab identity names the scaling, so it needs a lab
directory of its own, as another card count does; without the setting the identity, the logits and the speed are an
unscaled build's, byte for byte.

**A scaled build is measured, not held to the bar.** YaRN moves the model's own outputs (below), and llama.cpp's
backends part more under it — its CPU against its CUDA on code: 3.47e-4 scaled, 4.85e-4 not — so `exact` prints the
scaled comparison as "measured, not held to the bar" and fails only a port that is plainly broken: 10× the bar's KL,
5 points of top-1 under it, or a KL that is not a number. Byte for byte against the scaled build's own base still
judges, so `exact base` works for it as for any build.

**The ramp is checked on the CPU; the long prompt shows it.** On a ~1k-token prompt the dimensions YaRN rescales
hardly turn, so a wrong ramp scores the same there: at YaRN×2 the right one gave code 6.60e-4 and a ramp moved by 1.4
dimensions (`FNX_YARN_ORIG_CTX=131072`) 6.09e-4. On the 62k prompt they part: 1.88e-3 for the right one, 4.44e-3 for
the moved one (unscaled sFNX against unscaled llama.cpp there: 2.31e-3). `bin/rope_test` pins the band, every pair's
angle and the magnitude against ggml's formulas, the moved ramp as its counter-case.

| YaRN×2 against llama.cpp CUDA YaRN×2 | Top-1 agrees | KL mean |
| --- | --- | --- |
| Code, 300 steps | 99.00% | 6.60e-4 (llama.cpp CPU YaRN×2 against it: 3.47e-4) |
| Code edit, 300 steps | 100.00% | 5.10e-6 |
| 62k prompt, 200 steps | 97.50% | 1.88e-3 |
| The logo picture, 200 steps (teacher: `ref/img_logref` with the scaling) | 99.50% | 5.49e-3 (unscaled: 99.00% / 5.96e-3) |

Batched windows (`lab.sh batch`, YaRN×2) give each lane its tokens alone,
at the unscaled window times: six lanes 19.31 ms a window against 19.16, one lane 11.60 against 11.61. One run by hand
(no script in the repository): six concurrent 807k-token prompts at YaRN×4 and a 1,048,576 context each found a code
planted at their top, reads of 466–508 s (~1,650 tok/s) one after another.

**What scaling costs.** It changes every position's rotary, so outputs move at any length. llama.cpp with the scaling
against itself without, teacher-forced on the same tokens:

| | Code, top-1 / KL | Code edit, KL | 62k prompt, top-1 / KL |
| --- | --- | --- | --- |
| YaRN×2 | 99.67% / 8.2e-4 | 5.1e-6 | 98.0% / 7.2e-3 |
| YaRN×4 | 99.00% / 1.8e-3 | 5.0e-6 | 98.0% / 1.6e-2 |
| for scale: llama.cpp CPU against CUDA (62k: sFNX against llama.cpp) | 98.33% / 4.9e-4 | 4.1e-6 | 97.5% / 2.3e-3 |

The picks barely move; the distributions move by a few times the backends' own difference, more with S and with
length, and an edit not at all.

## Other GPUs

`ARCH=80`, `89`, `90`, `100` and `120` ([docs/config.md](config.md#other-gpus)) have never run: every object compiles
with nvcc 12.9 for each (2026-10-28, on prod's box), nothing more. To validate one, the steps under
[Other card counts](#other-card-counts) with `ARCH=n` on every make and a lab directory of its own, llama.cpp's
references made on the same cards; the start-up checks say first whether the cards and links can run it at all.

## RTX 3090s

`make ARCH=86` ([docs/config.md](config.md#other-gpus)) has run on a tester's box: four RTX 3090s on PCIe Gen4 x16, an
AMD EPYC, no NVLink, driver 595.84 (open kernel modules with the community P2P patch), CUDA 12.9, UD-Q4_K_XL with
int8 K/V and MTP drafts of 4. Their report, 2026-10-27/28, not measured here:

| Cards | Experts in RAM | Result |
| --- | --- | --- |
| 3 (`make ARCH=86`) | 28 GiB, batch 2 | loads in ~85–125 s; `serve/api_smoke.py` 10 of 10; ~178 tok/s one request, ~240 combined for two |
| 4 (`make ARCH=86 FNX_NCARD=4`) | 12 GiB, context 262,144 | loads in 88 s; greedy output the same as their three-card build; 163–170 tok/s on short generations, 152–200 for one request from 1k to 128k context |

Their prompts and sampling are not given, so these numbers do not line up with [docs/benchmarks.md](benchmarks.md).
Not run there: `lab.sh exact` against llama.cpp, `exact base`, `lab.sh batch` and `bin/sample_test`. Until they pass,
an sm_86 build's logits are unchecked, and four cards are as untested as [Other card counts](#other-card-counts) says.
To check one: the steps there with `ARCH=86` on every make and a lab directory of its own
(`FNX_LAB_DIR=/tmp/fnx-lab-$UID-sm86-${n}card`), llama.cpp's references made on the same cards.

What differs from a V100, as far as the code goes: the kernels are the same source, compiled for sm_86. The
`mma.sync.m8n8k4` in `src/moe_mma.cuh` runs there, but NVIDIA documents it as tuned for sm_70 and possibly much slower
on later cards. The hc read (`src/hc.cuh`) needs its 80 blocks resident at once: a 3090 has 82 SMs, and the engine
refuses as it loads a card with fewer or of another arch than the build's (`src/card_check.hpp`; `src/hc.cuh`
holds its grid to `HC_READ_BLOCKS` in `src/model.hpp` at compile time).

## The test inventory

| What | How |
| --- | --- |
| API surface through the front end (10 checks) | `serve/api_smoke.py` |
| Pictures through the front end (6 checks) | `serve/image_test.py` |
| Cancel, and K/V pool pressure on a 32k context (`serve/sfnx-pool.json`, arena on: 6 × 9k conversations, then 3 × 7k with 4,500-token copies). Prints the engine's paused / resumed / pool-full lines and asserts nothing itself — read the printout | `serve/cancel_pool_check.sh` |
| The arena: logits against every expert in VRAM, nothing lent and lending; batched = alone while lending | `tools/arena_check.sh` |
| Six ~160k-token conversations decoding at once | `serve/arena_long_check.sh` |
| Draft layer after batched windows in freshly lent pages (guards a NaN the draft layer met there) | `serve/gap_check.sh serve/sfnx-gap.json "plain="` |
| Batched windows = alone, with and without drafts (every token up to each reference's end) | `bin/batch_test`, `tools/lab.sh batch` |
| Batched drafts' sampler against plain sampling, in distribution (3 lanes, open1, 600 seeds: every p ≥ 0.34) | `SAMPLE_TEST_LANES=3 bin/sample_test` |
| The batched-draft row plan against a brute force over every split | `bin/batch_plan_test` |
| The prompt path's ring steps: two and three cards' sequences, and consecutive sums over both buffer sets on every supported count (CPU) | `bin/pf_ring_plan_test` |
| YaRN's band, per-pair angles and magnitude against ggml's formulas, and the settings it refuses | `bin/rope_test` |
| The card checks at load: a card's compute capability and SMs against the build's arch; PCI addresses; BAR1 from a V100's and an RTX 3090's sysfs text and one made from the 3090's at full ReBAR; the text it refuses, signs and over-wide fields included; which cards pass; the link probe's report (a pair left out refused), slow pairs warned and lost words named (CPU) | `bin/card_check_test` |
| The link probe on the cards (`src/peer_probe.cu`): its watcher's self-check and the healthy path run at every start, prod's included (the rates measured: [docs/config.md](config.md#boot-requirement)); its failure paths need a broken link and have run nowhere, checked by review only | engine start-up |
| The Makefile's checks of `ARCH` and `FNX_NCARD`: every bad value refused with its message before a stamp is written, a value padded with spaces the same setting (CPU) | `tools/build_guards_check.sh` |
| The top-k ties | `bin/topk_test` |
| The engine's stdin handling | `bin/sfnx_input_test` |
| The conversation cache where prompts share much: two conversations alternating on a lane, a system prompt changed near its end | `tools/lab.sh cmd cache2` |
| The front end's own tests | `.venv/bin/python -m pytest -q front/` (591 passed, 3 skipped without the anthropic SDK, ~4 min; against fake engines, no `bin/sfnx`) |

Bases for the exactness runs live in `/tmp/fnx-lab-$UID`, which a reboot empties: regenerate them with the previous
build before judging a new one (`tools/lab.sh exact base` writes one only when every prompt passes).

## The lab

For engine work the model stays loaded and each change costs seconds (`tools/lab.sh`).

- **The holder** (`bin/fnx_hold`) loads the model once: the dense weights on the cards and every routed expert's copy in
  host memory near its card, all from the CUDA virtual memory API at fixed addresses (`src/store.*`). It runs no kernel.
  46 s from the page cache; 7.5 GiB on the cards, 74 GiB of host copies (Q4).
- **A worker** (`bin/fnx_lab --attach`) maps that memory at the same addresses, takes the cards' tables over a socket,
  places the experts by its own settings, and runs commands from `build/libfnx_lab.so`: ~7 s to attach. A worker whose
  kernels fault ends, and the next command attaches a new one; the model is not loaded again.
- **Settings:** prod's are the defaults. `cmd "set VAR=..."` reloads the engine library (engine settings, at once);
  `FNX_ARENA`, `FNX_EXPERT_RAM_GIB` and `FNX_EXPERT_PROFILE` place the experts again in place (~3 s). A rebuilt library:
  `cmd reload`. A changed loader: `make lab`, `unload`, `start`.
- **Commands:** `read N...` (a prompt read's time, the GPUs sampled); `exact [NAME]` (logits on the 62k prompt, code1,
  edit1 and the picture through the prompt path, against the held model's llama.cpp references and a base;
  `exact base` writes the base); `decode [N] [K] [LK]` (one request alone as prod decodes it, edit1 and code1: tok/s and
  a round's verify / draft / commit time); `batch [GEN]` (batched windows give each lane its tokens alone, 1 to 6
  lanes); `compare A.lgr B.lgr`; `trace "CMD"` (that command under the CUPTI kernel tracer: each card's kernel totals
  and idle gaps, `make trace` builds the tracer); and every `fnx_run` mode (`rows`, `bench_step`, `mtp`, `spec`,
  `check`, `logits`, ...).
  A command's own setup: ~2 s for the engine (the arena filled from host memory by one kernel per card), ~1.2 s for its
  graphs.
- **Through prod:** `serve/solo_decode.sh` (edit1 and code1, three runs each, with the engine's own line per request)
  and `serve/pi_session.sh` (a short real pi session on `bench/pi-task`).
- **Debugging:** each command has a watchdog (`FNX_LAB_TIMEOUT`, default 600 s); `FNX_LAB_GDB=1` runs the worker under
  gdb. compute-sanitizer and `CUDA_LAUNCH_BLOCKING=1` cannot run the engine — they run one kernel at a time, and the
  ring sums' kernels wait on the other cards'. For a fault on the cards, the `check` mode launches without graphs.

The lab holds the model, so it and the service cannot share the box: stop `sfnx` before any GPU work, start it again
afterwards.
