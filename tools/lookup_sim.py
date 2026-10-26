"""Offline: tokens per verify round if the drafts came from the context (n-gram lookup) instead of, or before, the
draft layer, on a greedy run (the verify keeps the longest prefix of the drafts that matches greedy, plus one).

    lookup_sim.py PROMPT.bin GEN.txt [K] [NGRAM]

PROMPT.bin: make_prompt.py's file; GEN.txt: fnx_run gen's "generated: ..." line. Lookup: the latest earlier place
where the last NGRAM tokens occurred (prompt included), drafting the K tokens that followed it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
from refio import tokens  # noqa: E402

prompt = tokens(sys.argv[1])
gen = [int(t) for t in open(sys.argv[2]).read().split(":", 1)[1].split()]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 7
NG = int(sys.argv[4]) if len(sys.argv) > 4 else 3

ctx = list(prompt) + [gen[0]]
i = 1   # next generated token to produce
rounds = hits = 0
per_round = []
while i < len(gen):
    draft = []
    if len(ctx) > NG:
        key = ctx[-NG:]
        for s in range(len(ctx) - NG - 1, -1, -1):
            if ctx[s:s + NG] == key:
                draft = ctx[s + NG:s + NG + K]
                break
    acc = 0
    while acc < len(draft) and i + acc < len(gen) and draft[acc] == gen[i + acc]:
        acc += 1
    took = min(acc + 1, len(gen) - i)
    hits += draft != []
    ctx.extend(gen[i:i + took])
    i += took
    rounds += 1
    per_round.append(took)
print(f"{len(gen) - 1} tokens, {rounds} rounds: {(len(gen) - 1) / rounds:.2f} tokens per round with lookup alone "
      f"(K={K}, n-gram {NG}); a draft was found in {hits} rounds")
long = sum(1 for t in per_round if t >= 4)
print(f"rounds that took >= 4 tokens: {long} ({long * 100 // rounds}%), tokens in them: {sum(t for t in per_round if t >= 4)}")
