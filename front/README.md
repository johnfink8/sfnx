# front: sFNX's front end

The HTTP side of sFNX: OpenAI chat completions and responses, Anthropic messages, tools and strict tools, JSON
formats, pictures (through `bin/sfnx-vision`), the conversation cache's keep points, and the web app (Chat, Monitor,
About). It turns requests into token ids for `bin/sfnx` over its line protocol and streams the text back.

```bash
cd ~/src/sfnx
.venv/bin/python -m front.server --engine sfnx --config serve/sfnx.json      # what serve/run.sh and sfnx.service run
.venv/bin/python -m pytest -q front/                                          # 967 passed, 19 skipped (~6 min; no GPU)
```

The venv (`.venv`, Python 3.14) comes from `pyproject.toml` at the checkout's root, pinned to what prod runs:
`python3.14 -m venv .venv && .venv/bin/pip install '.[test]'` (or `uv venv -p 3.14 .venv && uv pip install -p .venv -r
pyproject.toml --extra test`). Add the `tools` extra (`numpy`, `pyyaml`) for the scripts in `ref/` and `tools/`.

## Where it came from

Copied on 2026-10-23 from Strata's `serve/` (github.com/Niko1221/Strata, MIT, `LICENSE` here) at `f487003a` on the
`responses-port-0221` branch (upstream `011c9e60` plus this box's Responses, literal-stop and batch-slot fixes), with
the Monitor tab's per-slot strip from that tree's working copy. Changes since: the package is `front` (not `serve`),
the engine choice is `--engine sfnx`, the web app has sFNX's name, logo and colours, and logs say `[sfnx]`.

`vendor/` holds the Strata modules the server imports, and nothing else of them: `strata_tokenizer.py` (the tokenizer
and its prompt encoder), `strata_control/` (the engine and vision launch arguments, the draft layer's rule, user
arguments, saved tunings, process records, paths), `strata_cli/client.py` (the old request shape the server still
accepts, and the tests' HTTP client), and two test helpers (`test_strata_tokenizer.py`, `make_profile.py`). Several
still carry branches for Strata's other platforms (Windows, AMD, Macs); they and the `STRATA_*` environment variables
the server reads are still to be trimmed and renamed.

The data the config names (`data/`, not in git except as noted): `data/tokenizer` (vocab, merges, token types and the
chat template, exported from the GGUF by Strata's tokenizer tool), `data/draft_vocab_en.bin` (the draft layer's
vocabulary) and `data/mtp-rt` (the draft layer's weights in the engine's format: Strata's `mtp_rt.py` files, plus
`experts-q4_0.bin`, its routed experts as Q4_0 made by `tools/mtp_experts.py q4_0` from the BF16 checkpoint tensors that
Strata's `mtp_fetch.py` saves; the engine reads it by default, `FNX_MTP_EXPERTS=q2_0` reads Strata's Q2_0 instead).
