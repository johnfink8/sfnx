#!/bin/sh
# Batched drafts by lane count: lab bdecode (lanes alternate prose2 and code1, prod's sampling, N tokens a lane) for
# LANES in 2..5, with FNX_BATCH_DRAFTS=1 and each lane's drafts fixed at 0..6 (FNX_BATCH_FIXED) and then the plan's;
# FNX_BATCH_DRAFTS=lookup with 0 drafts as one row a lane. Needs the lab started. Prints the bdecode lines.
#   tools/bdecode_sweep.sh [N]
cd "$(dirname "$0")/.." || exit 1
N=${1:-384}
lab() { tools/lab.sh cmd "$1" > /dev/null || { echo "FAILED: $1"; exit 1; }; }
run() {   # label lanes
    out=$(tools/lab.sh cmd "bdecode $2 $N") || { echo "FAILED: bdecode $2 ($1)"; exit 1; }
    echo "$out" | grep '^bdecode' | sed "s/^bdecode/$1:/"
}
for L in 2 3 4 5; do
    lab "set FNX_BATCH_DRAFTS=lookup"
    lab "set FNX_BATCH_FIXED=0"
    run "rows 1" "$L"
    lab "set FNX_BATCH_DRAFTS=1"
    for k in 0 1 2 3 4 5 6; do
        lab "set FNX_BATCH_FIXED=$k"
        run "fixed $k" "$L"
    done
    lab "unset FNX_BATCH_FIXED"
    run "plan" "$L"
done
