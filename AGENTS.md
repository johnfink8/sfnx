# sFNX development

sFNX is a from-scratch CUDA inference engine for one model (Qwen3.8-Flash-Next) on one kind of box: three Tesla
V100-32GB cards on PCIe Gen3 with no NVLink (sm_70, CUDA 12.9, gcc 14). It splits **every layer** across all three
cards (experts by FF width, recurrent heads, K/V by position, the vocabulary) so one request uses all three cards'
memory bandwidth at once, with a PCIe ring in place of NVLink, and serves the model through its own OpenAI/Anthropic
front end. `README.md` is the short version; `docs/` holds the design, the memory layout, the measurements,
the verification and the settings.

## Layout

- **`src/`**: the engine. CUDA kernels and headers (`.cuh`), `engine.cu`, `prefill.cu`, `model.cpp`, `session.cpp`,
  `batch.cpp` (batched windows with drafts), `batch_plan.hpp` (their row plan), `card_split.hpp` (how a layer
  divides over the cards, for any supported count), `pf_ring_plan.hpp` (the prompt path's ring steps),
  `card_check.hpp` (what a card must be for the build: arch, SMs, BAR1, checked as the model loads),
  `peer_probe.cu` / `.hpp` (every link written and watched before the engine uses it, and the report and failure text
  it prints). `kernels.cu`
  compiles as nine objects (`-DFNX_KPART=0..8`), each holding its share of the launchers.
- **Two models, one Engine** (`engine.hpp`): `engine_common.cu` holds what does not depend on the model (lanes and
  pages, windows over one lane or many, commits, saved states, the draft layer's window, the sampler's candidates, the
  window graphs), written against `engine_parts.hpp`; each model brings its per-card buffers and the few helpers that
  header names (`engine_impl.hpp` for Flash-Next, `s27_impl.hpp` for the 27B), its layers and embeddings (`engine.cu`,
  `s27_engine.cu`) and its prompt path (`prefill.cu`, `s27_prefill.cu`; their shared checks and ring conversions in
  `pf_common.cuh`). `owned.hpp` is the release list every engine registers on. A third model is a third header and
  engine file.
- **The 27B** (`make FNX_MODEL=27b`, Qwen3.8-27B dense, docs/config.md "Model"): `s27_attn.cuh` (head-split paged
  attention), `s27_ring3.cuh` (its sums); its differences in shared files (the loader, the kernels, the planner) sit
  in `#if FNX_27B` blocks. Its tools: `s27_run` (a prompt and greedy decode with drafts), `s27_ring_test` (its sums,
  timed and checked), `s27_probe` / `s27_bw_probe` (the weight-reading floor), `s27_cmp.py` (a layer dump against
  llama.cpp's), `s27_detok.py`, and `s27.sh` (runs any of them, and the checks, with the memlock limit).
- **`tools/`**: `sfnx.cpp` is the engine process (a line protocol on stdin, `sfnx_input.hpp`); `fnx_run.cpp` is the
  lab harness; `fnx_hold` / `fnx_lab` + `lab.sh` are the lab; the `*_test` programs and `*_check.sh` scripts are the
  checks (they share `labenv.sh`); the `*_probe` / `*_bench` programs measured the hardware the design rests on.
- **`front/`**: the Python serving layer (chat completions, responses, `/v1/messages`, tools, pictures in) and the web UI
  (`front/web/`). Ported from Strata's `serve/` (MIT, `front/LICENSE`).
- **`serve/`**: configs (`sfnx.json` is prod; `sfnx-q8.json`, `sfnx-gap.json`, `sfnx-pool.json` for the checks), the
  systemd unit, run scripts and served tests.
- **`vision/`**: the picture encoder process, on llama.cpp's mtmd.
- **`ref/`**: reference prompts (`*.bin`, token ids), the llama.cpp reference tools and their comparison scripts.
- **`data/`**: the tokenizer, the draft vocabulary, and `expert-routes.bin` (expert use tallied over an agentic
  code-review panel; it ranks which experts go to RAM).

## Who uses it, and what it may assume

The box serves a trusted LAN: pi sessions, review panels, one operator. **No auth, no sandbox, no untrusted input**:
the GGUFs, configs and prompts all come from the operator. Missing authentication or input validation is not a
finding; a hidden *failure* is.

The bar is correctness first: logits match llama.cpp to the same tolerance as llama.cpp's own CPU backend, on text and
pictures, at temperature 0, and in distribution when sampled with drafts. `docs/benchmarks.md` is the contract; a
change that silently moves them is a regression. Prod's settings are the code's defaults; environment settings exist
to try better ones, and a new behaviour lands off by default until the GPU checks below pass.

## Building

```sh
PATH=/usr/local/cuda/bin:$PATH make -j32 lab bin/sfnx   # ~2 min from scratch
make -j32 test-bin                                     # the test programs
make ref                                               # llama.cpp's reference tools
make trace                                             # the CUPTI tracer (tools/trace/) for lab.sh trace
```

- Every path in `serve/*.json`, the scripts and the unit is the checkout's (a config's relative `cwd` is the
  checkout, its other paths are `cwd`'s). The GGUFs come through `models`, a link to the folder holding
  `UD-Q4_K_XL/`, `Q8_0/` and `mmproj-F16.gguf`.
- llama.cpp at a37cc46 lives in `llama.cpp/` (a checkout or a link; `make LLAMA=path` for elsewhere), built with
  CUDA for the build's `ARCH` by the Makefile on first use. It supplies ggml headers, the picture encoder's libraries and the
  reference tools.
- `ARCH=86` builds for RTX 3090s and `CUDA=path` takes another toolkit (`docs/config.md`, "Other GPUs"); prod is the
  default 70. An sm_86 build has run only at a tester's, its logits unchecked (`docs/verification.md`, "RTX 3090s");
  80, 89, 90, 100 and 120 build and have never run; 75 is refused.
- The engine programs and the lab's library link one set of position-independent objects in `build/`.
- **Building writes `bin/sfnx` in place: stop the service first** (`sudo systemctl stop sfnx`), or build in another
  worktree.
- A change to the loader (`model.*`, `card_split.hpp`, `gguf.*`, `store.*`, `tools/lab_proto.hpp`) changes the lab's
  wire stamp: the running holder refuses new workers until `tools/lab.sh unload` + `start`.
- Made locally, not in git: `models` and `llama.cpp` (links), `data/mtp-rt/` (the draft layer's runtime files,
  `tools/mtp_experts.py` from the BF16 checkpoint), `ref/lgr/` (llama.cpp's logits, `make ref` then
  `ref/fnx_logref`), `.venv` (the front end's). A second worktree links all five from the main checkout.

## The GPUs: one user at a time

The service holds nearly all VRAM. **While it serves, nothing else may open a CUDA context**: no `bin/*` programs, no
`tools/lab.sh`, no `kbench` or `ncu`. Compiling, reading `/metrics` and logs, and CPU tests are fine. Stop the service
for GPU work and start it again when the work is done.

The lab (`tools/lab.sh`, usage at its top) holds the model once (`fnx_hold`) and runs `fnx_run`'s commands in a
worker, so a code change costs a library rebuild and `cmd reload`, not a model load. `set VAR=value` / `unset VAR`
change engine settings and are kept for replacement workers. `lab.sh cmd` (and so `exact`, `batch`, `decode`, `trace`)
exits with the worker's result code. The lab shows the build as many cards as it was made for (`build/ncard`). For
another count n (2 measured; 4, 6, 8 build but have never run, `docs/verification.md` says how to validate them):
`make FNX_NCARD=n lab` and a lab directory of their own (`FNX_LAB_DIR=/tmp/fnx-lab-$UID-${n}card` on every `lab.sh`
call), since their bases differ from three cards'; switching counts rebuilds everything.

Measure speed with `lab.sh decode` (edit1 and code1 greedy, prose2 and code1s with prod's sampling, seeds fixed) and
`lab.sh batch`; trace with `lab.sh trace`. A `kbench` win is not a win until the real window shows it. Sample GPU
utilization and power in every speed run. Mid perf push, measure speed only; run the exactness checks once the push is
done.

The 27B (`make FNX_MODEL=27b`, its own worktree: switching models rebuilds everything) has no lab: its checks load it
plain (~30 s) through `tools/s27.sh` (the memlock limit the loader needs, and this shell's `FNX_*` settings):
`tools/s27.sh bin/fnx_run models/Q8_0/Qwen3.8-27B-Q8_0.gguf exact` (and `decode`), `tools/s27.sh bin/batch_test MODEL
nextn data/draft_vocab_en.bin 64 ref/long2.bin ref/code1.bin ref/long2.bin`, `SAMPLE_TEST_LANES=3 tools/s27.sh
bin/sample_test MODEL nextn data/draft_vocab_en.bin ref/open1.bin 600 24`; serving through `serve/sfnx-27b.json`.
Its llama.cpp references (docs/verification.md, "The 27B") are made locally like the rest of `ref/lgr/`.

## The checks (on the GPU, before merging engine changes; the full inventory: `docs/verification.md`)

| Check | What it proves | How it fails |
| --- | --- | --- |
| `lab.sh exact` | logits within the bar against llama.cpp CUDA (code, edit, a 62k prompt, the logo picture), byte for byte against the base, 1 / 2 / 3 / 11 / 16-row windows identical to 5-row ones. Two cards' default fp16 sums are slightly over the bar on code and edit: judge a two-card build with `tools/lab.sh start "" "-E FNX_RING_F32=1"`. With `FNX_ROPE_SCALE` the llama.cpp comparison is against a scaled teacher and measured, not held to the bar (docs/verification.md, "Rope scaling") | `FAIL: ...`, exit 1 |
| `lab.sh exact base` | writes a new base, only when every prompt passes; `MODEL-base.id` records the build | `NOT written`, exit 1 |
| `lab.sh batch` | batched windows (with their drafts) give each lane its tokens alone | MISMATCH, exit 1 |
| `bin/sample_test` | speculative sampling matches plain sampling in distribution | MISMATCH, exit 1 |
| `bin/topk_test`, `bin/batch_plan_test`, `bin/sfnx_input_test`, `bin/pf_ring_plan_test`, `bin/rope_test`, `bin/card_check_test`, `tools/build_guards_check.sh` | the top-k ties, the batched row planner, the engine's stdin handling, the prompt path's ring steps on every supported count, YaRN's constants against ggml's, the card checks at load against real cards' numbers and sysfs, the link probe's report and failure text, the Makefile's checks of ARCH and FNX_NCARD (the last six CPU only) | non-zero exit |
| `serve/api_smoke.py`, `serve/image_test.py` | the API surface and pictures through the front end | `N failed` |
| `.venv/bin/python -m pytest -q front/` | the front end (CPU only) | pytest |

CI (`.github/workflows/ci.yml`, every PR and push to main, no GPU) builds `lab bin/sfnx test-bin` for sm_70 on every
card count, sm_86 on three and four and every other arch on three, and runs the CPU-only checks above and the front
end's tests (a skipped test fails it). The GPU rows stay the author's and the lead's to run.

Bases live in `/tmp/fnx-lab-$UID`, which a reboot empties: regenerate them with the previous build before judging a
new one.

## Deploying

The service runs from this checkout (`serve/sfnx.json`, `serve/sfnx.service.in`, port 8080; `make install` installs the
unit, `make install CONFIG=path` with another config, kept outside the checkout). After merging to main: stop the unit, `make -j32 lab bin/sfnx`, start it (~60 s warm), then
`serve/api_smoke.py http://127.0.0.1:8080`, `serve/image_test.py http://127.0.0.1:8080/v1` and
`serve/solo_decode.sh`. Rollback: the previous main commit, rebuilt the same way.

## Conventions

- Commits: one terse, informal line that hints at the move, no trailers.
- Code reads like the code around it: comments say why (with measurements), never history. Fail loudly: no silent
  defaults, skips or swallowed errors; a fused or fast path checks its preconditions on the host and throws outside
  the shapes it was compiled for.
- No `.bak` copies (git is the rollback). Never `rm` a path built from a variable: delete by literal path (`unlink`
  for one file). Never `pkill -f` with the pattern on your own command line.
- Parallel agents work in separate git worktrees on their own branches, each told which files the others own; the
  lead merges, runs the checks above, and only then moves master and restarts the service.
- Docs: plain words and measured numbers, with what they were measured on. Nothing new named "strata".
