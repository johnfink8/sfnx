# Contributing to sFNX

sFNX is built for three PCIe V100s and judged there: logits within llama.cpp's own CPU tolerance at temperature 0,
batched lanes giving each lane its tokens alone, and the speeds in `docs/benchmarks.md`. A change that moves any of
those without saying so is a regression, however fast. Ports to other cards and card counts are welcome; say what ran
where.

- **`AGENTS.md`** is the working guide: layout, building, the lab, the checks, the conventions. It is written for
  agents and people alike.
- **`docs/verification.md`** is every check, what it proves and how it fails, and how to validate another card count or
  GPU. **`docs/config.md`** is every setting.

## A pull request

1. Branch from `main`. Keep a PR to one move.
2. Build and run the checks your change can affect on whatever cards you have (`AGENTS.md`, "The checks"). CI builds
   the engine for every card count and arch (not the picture encoder), runs the CPU checks and the front end's tests;
   it has no GPU, so a green CI says nothing about the logits or speed.
3. Fill in the template: the cards it ran on, each check run or not run and why, speed against `main` on the same
   cards for engine changes. A check you could not run is fine to say; a check you did not run is not fine to imply.
4. The maintainer runs the GPU checks again on the three V100s before merging.

## What gets merged

- **Prod's behaviour stays the default.** A new behaviour lands behind a setting, off, until the GPU checks pass.
- **Fail loudly.** No silent defaults, skips or swallowed errors. A fused or fast path checks its preconditions on the
  host and throws outside the shapes it was compiled for.
- **Measured claims.** Docs and comments give numbers with what they were measured on. Comments say why, never history.
- **Commits:** one short, informal line that hints at the move.

## Reporting

Use the issue forms: **Bug**, **Other cards** (a run on cards, counts or links other than three PCIe V100s, working or
not), or **Idea**. Security: `SECURITY.md`.
