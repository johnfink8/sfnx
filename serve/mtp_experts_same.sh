#!/bin/bash
# The greedy tokens of code1 and edit1 (1,024 each, K=4, lookup 15) with each FNX_MTP_EXPERTS, in the lab: the draft
# layer's experts may change speed, never the tokens. Leaves FNX_MTP_EXPERTS unset.
cd "$(dirname "$0")/.." || exit 1
out=${1:-/tmp/fnx-mtp-same-$(id -u)}
mkdir -p "$out"
for q in q2_0 q4_0; do
    timeout 120 tools/lab.sh cmd "set FNX_MTP_EXPERTS=$q" >/dev/null || exit 1
    for p in code1 edit1; do
        timeout 600 tools/lab.sh cmd "mtp $PWD/data/mtp-rt $PWD/ref/$p.bin 1024 4 $PWD/data/draft_vocab_en.bin 15" \
            | grep '^generated:' | tr ' ' '\n' | sed -n 2,1025p >"$out/$p-$q.txt"   # the last round may run past 1,024
        [ -s "$out/$p-$q.txt" ] || { echo "$p $q: no generated line"; exit 1; }
    done
done
timeout 120 tools/lab.sh cmd "unset FNX_MTP_EXPERTS" >/dev/null
rc=0
for p in code1 edit1; do
    if cmp -s "$out/$p-q2_0.txt" "$out/$p-q4_0.txt"; then
        echo "$p: same $(wc -l <"$out/$p-q2_0.txt") tokens"
    else
        echo "$p: DIFFERENT"; rc=1
    fi
done
exit $rc
