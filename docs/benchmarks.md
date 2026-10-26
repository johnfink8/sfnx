# Benchmarks

Every speed number here was measured on the reference box — three Tesla V100-32GB cards on PCIe Gen3, no NVLink — with
the build and config it came from. Correctness against llama.cpp is
[docs/verification.md](verification.md).

## How these are measured

- **Through the front end** unless a row says the lab. Requests go to the served engine over HTTP via
  `tools/strata_stream.py`, so tokenizer, streaming and the front end's own work are inside the number.
- **Greedy** unless a row gives a sampling setting. `ref/edit1` is a code-edit prompt (636 tokens), `ref/code1` new
  code (1,024), `ref/prose2` prose reasoning.
- **Solo decode** (`serve/solo_decode.sh`) runs edit1 and code1 three times each against prod, prose2 five times since
  it is sampled, and prints the engine's own line per request. **The lab** (`tools/lab.sh decode`) runs the same
  prompts with seeds fixed.
- **GPU utilization** is the mean over the run of all three cards, sampled during the run.
- A range is the spread across runs, not a confidence interval.

Decode speed is tokens a round over the round's time. One verify window checks the drafts (1 to 6, as many as pay at
the request's measured acceptance: a 3-row window ~13 ms, 5 rows ~16) or up to 15 from n-gram lookup (~28 ms with 16),
and costs the same however many survive. Code, and edits above all, draft well; prose drafts poorly — 47% of drafts
accepted at prod's sampling, 34% at temperature 1.0. That difference is most of the spread in the tables below.

## One request, Q4, prod's config

`serve/sfnx.json`: int8 K/V and the arena, `ref/*.bin` through `front/`.

| Workload | tok/s | Per round |
| --- | --- | --- |
| Code edit (`ref/edit1`), greedy | 366.9–381.6 | 11.0 tokens, ~29.5 ms |
| New code (`ref/code1`), greedy | 200.4–229.1 | 3.8–4.6 tokens |
| Prose reasoning (`ref/prose2`), temperature 0.6 | 134.4–140.8 | 2.1 tokens, ~15 ms |
| Prose sampled, temperature 1.0, 1k-word prompt | 138.6–139.6 steady | GPUs 92–93% |
| A short pi session (`serve/pi_session.sh`: pi fixes a bug and writes tests, 7–10 requests, 3k–9k-token prompts, sampled) | 168.7–176.4 | 3.2–3.4 tokens, ~19 ms |

## Several conversations at once

Throughput here is **output tokens a second summed over all the lanes**, with the per-conversation figure in brackets.
Six requests of 1,000 random words, 1,000 tokens each, prose at temperature 1.0, steady part:

| | tok/s combined | each |
| --- | --- | --- |
| Batched drafts (prod) | 280.3–290.9, GPUs 90% | ~47 |
| `FNX_BATCH_DRAFTS=0`, one row a lane | 303.0–303.9 | ~51 |

Six lanes of temperature-1.0 prose is the one load where batched drafts lose: their drafts stand ~70% of the time at
one a lane, and the plan's cost model runs ~8% low on wide windows. Elsewhere it wins — 7–9% more output than one row a
lane on the review-panel replay (296 against 318 s), and 4–43% ahead of any fixed count of drafts a lane on mixed
traffic. See [docs/design.md](design.md).

The same six lanes when each conversation carries ~160k tokens of context — 960k positions of K/V, read in one at a
time and then decoded together (`serve/arena_long_check.sh`, 400 tokens a conversation): **237.6 tok/s combined, ~40
a conversation**. Lower than the short-context number because attention runs over 960k positions instead of 6k, and
holding them displaces experts: the pool lends 4.5 GiB of card 2's expert memory, and lent experts cost a batch 3.9–5.1%
([docs/memory.md](memory.md)).

## Prompts and first tokens

| Workload | sFNX | With 1,024-row prompt chunks |
| --- | --- | --- |
| 30k random words, cold read (`serve/prompt_read.py`, 31,965 tokens) | 3,667 tok/s | 2,047 |
| 114k-token prompt of source code, cold read (`serve/long_prompt.py`) | 3,373 tok/s (33.8 s), GPUs 93–97% | 1,855 |
| After that 114k prompt, decode | 135.4 tok/s (150.1 on the reference's own tokens) | |
| Warm turn: +2k tokens on a 30k conversation, first token (`serve/warm_ttft.py`) | 0.95 s median | |

The prompt-path work behind those two reads: the attention exchange in pieces with its merge deferred, the expert
kernel on raw `mma.m8n8k4`, the recurrent core in chunks, attention tiles of 32 positions, QSA in groups, and K/V by
thirds. In the lab, a 30k read went 3,176 → 3,720 tok/s.

## Whole sessions, replayed

| Workload | sFNX |
| --- | --- |
| A code-review panel's first round (5 reviewers, 83 calls, prompts up to ~48k tokens, the recorded replies' lengths, two runs) | 264–271 s; 154.5–168.5 tok/s over the run; first token median 1.0 s; GPUs 90% |
| A pi reviewer session (code-style reviewer's first round, 14 calls; its first call an 11k real prompt) | 96 s; 125.6 tok/s over the run |

## Against this box's previous engine

sFNX as prod runs it, against Strata as this box's service ran it before the switch: the same model file,
`--batch 6 --batch-groups 3`, MTP drafts, 94% of the experts in VRAM. **These are our measurements of one Strata
config on our hardware at one point in time** — not a statement about that engine today, and not a comparison anyone
else should reproduce on different parts.

| Workload | sFNX | Previous engine |
| --- | --- | --- |
| Code-review panel's first round, replayed | **264–271 s**, 154.5–168.5 tok/s | 484 s, 76.5 tok/s |
| A pi reviewer session, replayed | **96 s**, 125.6 tok/s | 228 s |
| Six requests decoding at once (combined) | **280.3–290.9** with batched drafts; 303.0–303.9 without | 198.7 tok/s |
| One request, prose sampled at temperature 1.0 | **138.6–139.6** steady | 68.1 tok/s |
| Warm turn, first token | **0.95 s** median | 2.07 s |
| 30k random words, cold read | **3,667 tok/s** | 2,454 tok/s |
| 114k prompt of source code, cold read | **3,373 tok/s** (33.8 s), GPUs 93–97% | 2,971 tok/s (38.4 s), GPUs ~73% |
| After that 114k prompt, decode | **135.4 tok/s** | 72.9 tok/s |
| Code-edit and new-code decode, prose reasoning, a short pi session | as the tables above | not measured |

## Q8_0

`serve/sfnx-q8.json` at 52 GiB of experts in RAM and lookup drafts capped at 7, against the same engine on the same
files with its prod settings. Settings and the arena side of Q8: [docs/memory.md](memory.md).

| Workload | sFNX Q8_0 | Previous engine |
| --- | --- | --- |
| One request (1k-word prompt, 1,000 tokens) | 88.9 tok/s — taken before the prompt passes below, so a floor | 53.1 tok/s |
| Five at once, same (steady / over the run / first token median) | 180.3 / 123.8 tok/s / 7.7 s | 143.2 / 79.4 tok/s / 20.1 s |
| New code (`ref/code1.bin`) through the API, temperature 0, three runs | 116.4 / 122.3 / 118.7 tok/s | 59.7 / 67.5 / 69.3 tok/s |
| A code edit (`ref/edit1.bin`), same | 139.9 / 146.2 / 144.4 tok/s | 63.6 / 76.9 / 79.5 tok/s |

Both start in ~310 s. Lab decode at 58 GiB with lookup drafts up to 15: code edit 137.8 tok/s (130.4 with lookup capped
at 7), new code 111.5.

## The reference implementation on the same cards

Upstream llama.cpp (a37cc46, same model, all layers on the three cards, `llama-bench`): 46.9 tok/s decode and 439 tok/s
at pp512 with a layer split; 35.1 and 162 with its tensor split.

## Numbers this repo no longer publishes

Earlier revisions compared the box against other engines and other owners' rigs — figures taken from posts, issue
threads and model cards, on prompts and hardware we cannot see. They are gone: they move under us, we cannot re-run
them, and a table that ranks someone else's engine on our prompts is not ours to publish. `git log -p README.md` has
them with their links, if a future comparison needs them again.
