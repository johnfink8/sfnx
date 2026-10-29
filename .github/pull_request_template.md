<!-- The rules are in CONTRIBUTING.md and AGENTS.md; the checks, and how each fails, in docs/verification.md. -->

## What and why

<!-- What changes, and the measurement or failure that motivated it. Link the issue if there is one. -->

## Cards it ran on

<!-- How many cards, which (compute capability), the link between them (PCIe gen, NVLink), driver, CUDA, and the
     build (ARCH, FNX_NCARD). "None" if nothing here ran on a GPU. -->

## Checks

<!-- CI builds the engine for every card count and arch (not the picture encoder) and runs the CPU checks and the
     front end's tests; it has no GPU. The GPU checks are yours to run on your cards; the maintainer runs them again on the three V100s before merging. Tick
     what you ran and paste its last lines below; for what you did not run, say why. -->

- [ ] `tools/lab.sh exact` (logits against llama.cpp, and byte for byte against main's base)
- [ ] `tools/lab.sh batch`
- [ ] `bin/sample_test`
- [ ] `bin/topk_test`
- [ ] `serve/api_smoke.py`, `serve/image_test.py` (for changes to the front end or the API)

## Speed

<!-- For engine changes: tools/lab.sh decode (and batch, if batched windows are touched), main against this branch on
     the same cards, with GPU utilization and power sampled. "Not measured" otherwise. -->

## Defaults

<!-- Does anything change for a config that sets nothing new? New behaviour lands off by default until the GPU checks
     pass. -->
