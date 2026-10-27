#!/bin/sh
# The lab: the model held once by bin/fnx_hold (unit fnx-hold: the weights in shared memory, no kernels), commands run
# by a worker attached to it (bin/fnx_lab --attach, unit fnx-lab) from build/libfnx_lab.so. A worker that dies (a fault
# in a kernel) is replaced by the next command in seconds; the model is loaded again only by `unload` + `start` or a new
# MODEL. Both run prod's settings, the engine's defaults; variables override them.
#   tools/lab.sh start [MODEL.gguf] ["-E VAR=value ..."]   the holder if it is not holding MODEL (default Q4), then a
#                                                          worker with these variables (kept for its replacements)
#   tools/lab.sh cmd "read 30720"     a fnx_run command, or set/unset VAR (engine settings at once; FNX_ARENA,
#                                     FNX_EXPERT_RAM_GIB, FNX_EXPERT_PROFILE place the experts again, ~seconds), reload
#   tools/lab.sh read N [N...]        the first N tokens of ref/long64.bin through the prompt path: time, tok/s, the GPUs
#   tools/lab.sh exact [NAME]         logits on ref/long64, code1 and edit1 (prompt path + the reference's steps)
#                                     against llama.cpp and against $DIR/MODEL-base[-PROMPT].lgr, both the held
#                                     model's ("exact base" writes the baselines); see fnx_run.cpp lab_check
#   tools/lab.sh batch [GEN]          batched windows against each conversation alone (tools/batch_check.hpp): 1, 2,
#                                     3 and 6 lanes, GEN greedy tokens each (default 64)
#   tools/lab.sh decode [N] [K] [LK] [SAMPLES]  one request alone as prod decodes (MTP drafts up to K=6, lookup up to
#                                     15): edit1 and code1 greedy (N tokens, default 1,024), prose2 (thinking-style
#                                     prose) with prod's sampling, seeds 1-5 summed: tok/s, tokens and ms a round.
#                                     SAMPLES: some of edit1,code1,prose2 (default all)
#   tools/lab.sh trace "read 30720"   the command traced (CUPTI): each card's kernel totals and idle gaps
#   tools/lab.sh stop                 the worker (the holder keeps the model)
#   tools/lab.sh unload               the worker and the holder
# make build/libfnx_lab.so between commands, then "reload" (set/unset reload by themselves); a change to the loader
# (model.*, gguf.*, store.*, lab_proto.hpp) needs `make lab`, then unload + start: the holder refuses a worker built
# from other loader sources. cmd and the checks exit with the worker's result (0 passed, 1 failed, 70 a broken worker).
cd "$(dirname "$0")/.." || exit 1
DIR=${FNX_LAB_DIR:-/tmp/fnx-lab-$(id -u)}
FIFO=$DIR/cmd.fifo
LOG=$DIR/out.log
HLOG=$DIR/hold.log
SOCK=$DIR/hold.sock
Q4=$(pwd)/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
active() { [ "$(systemctl is-active "$1")" = active ]; }
env_record() {   # a set/unset the worker took, kept in worker.env: a replacement worker and trace's start with it too
    var=${1#* }
    var=${var%%=*}
    out="" prev=""
    for w in $(cat "$DIR/worker.env" 2> /dev/null); do
        if [ "$prev" = "-E" ]; then
            case "$w" in "$var="*) ;; *) out="$out -E $w" ;; esac
        elif [ "$w" != "-E" ]; then
            out="$out $w"
        fi
        prev=$w
    done
    case "$1" in "set "*) out="$out -E ${1#set }" ;; esac
    echo "$out" > "$DIR/worker.env"
}
unit() {   # name log env... -- command...
    name=$1 log=$2
    shift 2
    sudo systemctl stop "$name" 2> /dev/null
    sudo systemctl reset-failed "$name" 2> /dev/null
    sudo systemd-run --unit="$name" -p LimitMEMLOCK=infinity --uid="$(id -u)" --gid="$(id -g)" -p WorkingDirectory="$(pwd)" \
        -p StandardOutput=append:"$log" -p StandardError=append:"$log" -E FNX_LAB_DIR="$DIR" "$@" > /dev/null 2>&1
}
wait_for() {   # unit log pattern tries (a fifth of a second each)
    for _ in $(seq "$4"); do
        grep -q "$3" "$2" && { grep "$3" "$2" | tail -1; return 0; }
        active "$1" || { echo "$1 ended:"; tail -20 "$2"; return 1; }
        sleep 0.2
    done
    echo "$1: no '$3' yet"; tail -5 "$2"; return 1
}
hold() {   # model
    if active fnx-hold && [ "$(cat "$DIR/hold.model" 2> /dev/null)" = "$1" ]; then return 0; fi
    sudo systemctl stop fnx-lab 2> /dev/null
    # No warm-up read: the loader reads the files at the drive's rate itself (src/gguf.hpp Stager), the pieces in
    # the page cache through it and the rest past it. The estimate: the part not in the page cache at the last cold
    # load's disk rate (the loader prints it), else the NVMe's 3.4 GB/s (tools/stage_test.cpp, 2026-10-25); the host
    # copies and the uploads come on top
    # shellcheck disable=SC2046
    shards=$(ls $(dirname "$1")/$(basename "$1" | sed 's/-00001-of-/-0000?-of-/') 2> /dev/null || echo "$1")
    # shellcheck disable=SC2086
    cold=$(fincore -b -n -o RES,SIZE $shards | awk '{c += $2 - $1} END {printf "%.0f", c / 1e9}')
    rate=$(cat "$DIR/disk.rate" 2> /dev/null || echo 3.4)
    last=$(cat "$DIR/hold.last" 2> /dev/null)
    [ "$cold" -gt 1 ] && echo "($cold GB of the model are not in the page cache: ~$(awk -v c="$cold" -v r="$rate" 'BEGIN {printf "%.0f", c / r}') s to read them at $rate GB/s${last:+; the last load: $last})"
    : > "$HLOG"
    echo "$1" > "$DIR/hold.model"
    unit fnx-hold "$HLOG" -- "$(pwd)/bin/fnx_hold" "$1" "$SOCK"
    wait_for fnx-hold "$HLOG" "^== holding" 6000 || return 1
    # for the next estimate: the disk rate when the disk gave nearly all of the weights, and how long this load took
    r=$(sed -n 's/.*GB from the disk at \([0-9.]*\) GB\/s.*/\1/p' "$HLOG" | tail -1)
    [ -n "$r" ] && echo "$r" > "$DIR/disk.rate"
    t=$(sed -n 's/^== holding: .* in \([0-9.]*\) s .*/\1/p' "$HLOG" | tail -1)
    [ -n "$t" ] && echo "$t s with $cold GB not in the page cache" > "$DIR/hold.last"
    return 0
}
worker() {   # a worker with the saved variables
    [ -p "$FIFO" ] || mkfifo "$FIFO"
    [ -s "$LOG" ] && mv -f "$LOG" "$LOG.prev"   # the last worker's output (how it ended) stays readable
    : > "$LOG"
    # FNX_LAB_GDB=1: the worker under gdb, so a crash on the host leaves its backtrace in the log
    # (compute-sanitizer cannot run the engine: it runs one kernel at a time, and the ring sums' kernels wait on the
    # other cards' kernels, so the first one spins for ever. For a fault on the cards: -E CUDA_LAUNCH_BLOCKING=1 at
    # start and the check mode, which launches without graphs)
    gdb=""
    [ "${FNX_LAB_GDB:-0}" = 1 ] && gdb="/usr/bin/gdb -q -batch -ex run -ex bt -ex quit --args"
    # shellcheck disable=SC2046,SC2086
    unit fnx-lab "$LOG" $(cat "$DIR/worker.env" 2> /dev/null) -- $gdb "$(pwd)/bin/fnx_lab" --attach "$SOCK" "$FIFO" "$(pwd)/build/libfnx_lab.so"
    wait_for fnx-lab "$LOG" "^== ready" 1500
}
case "$1" in
start)
    mkdir -p "$DIR"
    echo "${3:-}" > "$DIR/worker.env"
    hold "${2:-$Q4}" && worker
    ;;
cmd)
    active fnx-hold || { echo "the lab is not started (tools/lab.sh start)"; exit 1; }
    # worker.env is read back by word splitting (a value with a space would become two arguments)
    case "$2" in "set "*" "* | "set "*"	"*) echo "set: a value with a space or tab cannot be kept for replacement workers" >&2; exit 2 ;; esac
    # a worker that ends between the check and the write leaves the pipe without a reader: the write times out, a new
    # worker attaches and the command goes again
    for try in 1 2; do
        if ! active fnx-lab; then
            echo "(a new worker: the last one ended)"
            worker > /dev/null || exit 1
        fi
        from=$(stat -c %s "$LOG")
        before=$(grep -c "^== done" "$LOG")
        timeout 5 sh -c 'echo "$1" > "$2"' _ "$2" "$FIFO" && break
        [ "$try" = 2 ] && { echo "the worker does not read its pipe"; exit 1; }
        sudo systemctl stop fnx-lab
    done
    # a command that hangs (a kernel spinning on the cards) is ended after FNX_LAB_TIMEOUT seconds with its worker
    deadline=$(($(date +%s) + ${FNX_LAB_TIMEOUT:-600}))
    while [ "$(grep -c "^== done" "$LOG")" -le "$before" ]; do
        if [ "$(date +%s)" -gt "$deadline" ]; then
            tail -c +$((from + 1)) "$LOG" | tail -20
            nvidia-smi --query-gpu=index,utilization.gpu,power.draw --format=csv,noheader
            echo "(no answer in ${FNX_LAB_TIMEOUT:-600} s: the worker is ended; the next command gets a new one)"
            sudo systemctl stop fnx-lab
            exit 124
        fi
        # a worker that ends right after its "== done" line (a broken context: 70) is not a death without an answer
        if ! active fnx-lab && [ "$(grep -c "^== done" "$LOG")" -le "$before" ]; then
            tail -c +$((from + 1)) "$LOG"
            echo "(the worker died; a new one attaches)"
            worker > /dev/null && grep "^== ready" "$LOG"
            exit 1
        fi
        sleep 0.2
    done
    out=$(tail -c +$((from + 1)) "$LOG")
    printf '%s\n' "$out"
    # the command's own result: the worker's "== done N" (0 passed, 1 failed, 70 its CUDA contexts broke)
    rc=$(printf '%s\n' "$out" | sed -n 's/^== done \([0-9]*\).*/\1/p' | tail -1)
    [ -n "$rc" ] || { echo "(no result line from the worker)"; exit 1; }
    case "$2" in
    "set "* | "unset "*) [ "$rc" = 0 ] && env_record "$2" ;;
    esac
    exit "$rc"
    ;;
read | exact | batch | decode)
    # the standard checks (fnx_run.cpp lab_check), the GPUs sampled while they run
    nvidia-smi --query-gpu=index,utilization.gpu,power.draw --format=csv,noheader,nounits -lms 500 > "$DIR/gpu.csv" &
    smi=$!
    "$0" cmd "$*"
    rc=$?
    kill $smi
    awk -F', ' '{u[$1] += $2; w[$1] += $3; n[$1]++} END {s = "GPUs:"; for (c = 0; c < 3; c++) if (n[c]) s = s sprintf(" %d: %.0f%% %.0f W", c, u[c] / n[c], w[c] / n[c]); print s}' "$DIR/gpu.csv"
    exit $rc
    ;;
trace)
    # the command on a worker with the kernel tracer (tools/trace/libcupti_trace.so, `make trace`), which then quits so the tracer's
    # exit flush writes everything; each card's kernel totals and idle gaps (a read's: from after warm_graphs'
    # windows); then a plain worker again
    P=$(pwd)/tools/trace
    saved=$(cat "$DIR/worker.env" 2> /dev/null)
    rm -f "$DIR"/trace.*.csv
    echo "$saved -E CUDA_INJECTION64_PATH=$P/libcupti_trace.so -E CUPTI_TRACE_OUT=$DIR/trace -E LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64" > "$DIR/worker.env"
    sudo systemctl stop fnx-lab
    "$0" cmd "$2"
    rc=$?
    echo "$saved" > "$DIR/worker.env"
    timeout 30 sh -c 'echo quit > "$1"' _ "$FIFO"
    for _ in $(seq 150); do active fnx-lab || break; sleep 0.2; done
    t=$(ls -t "$DIR"/trace.*.csv 2> /dev/null | head -1)
    [ -n "$t" ] || { echo "no trace written"; exit 1; }
    read_only=0
    case "$2" in read*) read_only=1 ;; esac
    for d in 0 1 2; do
        READ=$read_only python3 "$P/pf_kernels.py" "$t" "$d" 14
        READ=$read_only python3 "$P/pf_gaps.py" "$t" "$d" 8
    done
    echo "(trace: $t)"
    worker > /dev/null
    exit $rc
    ;;
stop)
    sudo systemctl stop fnx-lab
    ;;
unload)
    sudo systemctl stop fnx-lab fnx-hold
    ;;
*)
    echo "usage: tools/lab.sh start [MODEL] [\"-E VAR=value ...\"] | cmd \"COMMAND\" | read N... | exact [NAME] | batch [GEN] | decode [N] [K] [LK] [SAMPLES] | trace \"COMMAND\" | stop | unload" >&2
    exit 2
    ;;
esac
