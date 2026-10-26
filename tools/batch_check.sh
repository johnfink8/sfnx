#!/bin/sh
# Batched windows against each conversation alone (bin/batch_test, Q4): lanes holding long prompts (past 2,048
# positions, so the QSA selection runs) and a short one; then the window cost by lanes (fnx_run bench_step, steps only).
# Exits 1 when a lane's tokens differ from its conversation's alone (batch_test's MISMATCH) or a run fails.
#   tools/batch_check.sh [GEN]
GEN=${1:-64}
. "$(dirname "$0")/labenv.sh"
lab_tmp
echo "== batch_test, $GEN tokens"
run batch -E FNX_MAX_CTX=100000 -- bin/batch_test "$G" "$FNX/data/mtp-rt" data/draft_vocab_en.bin "$GEN" \
    ref/long2.bin ref/code1.bin ref/long2.bin
grep -vE "^\[|^fnx: |loaded" "$O/batch.log"
echo "== bench_step, 16k context"
run bench_step -E FNX_MAX_CTX=104448 -E FNX_LANES=6 -- bin/fnx_run "$G" bench_step ref/long64.bin 6 40 16384
grep -E "^(verify|step|fnx_run)" "$O/bench_step.log"
finish
