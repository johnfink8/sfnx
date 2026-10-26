# sFNX development

sFNX is a from-scratch CUDA inference engine for one model (Qwen3.8-Flash-Next) on one kind of box: three Tesla
V100-32GB cards on PCIe Gen3 with no NVLink (sm_70, CUDA 12.9, gcc 14). It splits **every layer** across all three
cards (experts by FF width, recurrent heads, K/V by position, the vocabulary) so one request uses all three cards'
memory bandwidth at once, with a PCIe ring in place of NVLink, and serves the model through its own OpenAI/Anthropic
front end. `README.md` is the design and the measurements.

## Layout

- **`src/`**: the engine. CUDA kernels and headers (`.cuh`), `engine.cu`, `prefill.cu`, `model.cpp`, `session.cpp`,
  `batch.cpp` (batched windows with drafts), `batch_plan.hpp` (their row plan). `kernels.cu` compiles as nine objects
  (`-DFNX_KPART=0..8`), each holding its share of the launchers.
- **`tools/`**: `sfnx.cpp` is the engine process (a line protocol on stdin, `sfnx_input.hpp`); `fnx_run.cpp` is the
  lab harness; `fnx_hold` / `fnx_lab` + `lab.sh` are the lab; the `*_test` programs and `*_check.sh` scripts are the
  checks (they share `labenv.sh`); the `*_probe` / `*_bench` programs measured the hardware the design rests on.
- **`front/`**: the Python serving layer (chat completions, responses, `/v1/messages`, tools, images) and the web UI
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
pictures, at temperature 0, and in distribution when sampled with drafts. The README's numbers are the contract; a
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
  CUDA for sm_70 by the Makefile on first use. It supplies ggml headers, the picture encoder's libraries and the
  reference tools.
- The engine programs and the lab's library link one set of position-independent objects in `build/`.
- **Building writes `bin/sfnx` in place: stop the service first** (`sudo systemctl stop sfnx`), or build in another
  worktree.
- A change to the loader (`model.*`, `gguf.*`, `store.*`, `tools/lab_proto.hpp`) changes the lab's wire stamp: the
  running holder refuses new workers until `tools/lab.sh unload` + `start`.
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
exits with the worker's result code.

Measure speed with `lab.sh decode` (edit1 and code1 greedy, prose2 and code1s with prod's sampling, seeds fixed) and
`lab.sh batch`; trace with `lab.sh trace`. A `kbench` win is not a win until the real window shows it. Sample GPU
utilization and power in every speed run. Mid perf push, measure speed only; run the exactness checks once the push is
done.

## The checks (on the GPU, before merging engine changes)

| Check | What it proves | How it fails |
| --- | --- | --- |
| `lab.sh exact` | logits within the bar against llama.cpp CUDA (code, edit, a 62k prompt, the logo picture), byte for byte against the base, 1 / 2 / 3 / 11 / 16-row windows identical to 5-row ones | `FAIL: ...`, exit 1 |
| `lab.sh exact base` | writes a new base, only when every prompt passes; `MODEL-base.id` records the build | `NOT written`, exit 1 |
| `lab.sh batch` | batched windows (with their drafts) give each lane its tokens alone | MISMATCH, exit 1 |
| `bin/sample_test` | speculative sampling matches plain sampling in distribution | MISMATCH, exit 1 |
| `bin/topk_test`, `bin/batch_plan_test`, `bin/sfnx_input_test` | the top-k ties, the batched row planner, the engine's stdin handling (the last two CPU only) | non-zero exit |
| `serve/api_smoke.py`, `serve/image_test.py` | the API surface and pictures through the front end | `N failed` |
| `.venv/bin/python -m pytest -q front/` | the front end (CPU only) | pytest |

Bases live in `/tmp/fnx-lab-$UID`, which a reboot empties: regenerate them with the previous build before judging a
new one.

## Deploying

The service runs from this checkout (`serve/sfnx.json`, `serve/sfnx.service.in`, port 8080; `make install` installs the
unit). After merging to master: stop the unit, `make -j32 lab bin/sfnx`, start it (~60 s warm), then
`serve/api_smoke.py http://127.0.0.1:8080`, `serve/image_test.py http://127.0.0.1:8080/v1` and
`serve/solo_decode.sh`. Rollback: the previous master commit, rebuilt the same way.

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
