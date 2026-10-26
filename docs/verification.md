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

- The sums between cards travel as fp16 pairs. `FNX_RING_F32=1` sends them as fp32: KL 5.75e-4 → 4.38e-4 on the code
  prompt.
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
| The top-k ties | `bin/topk_test` |
| The engine's stdin handling | `bin/sfnx_input_test` |
| The conversation cache where prompts share much: two conversations alternating on a lane, a system prompt changed near its end | `tools/lab.sh cmd cache2` |
| The front end's own tests | `.venv/bin/python -m pytest -q front/` (967 passed, 19 skipped, ~6 min; against fake engines, no `bin/sfnx`) |

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
