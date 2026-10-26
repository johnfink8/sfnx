# Sourced by the lab's GPU scripts (`. "$(dirname "$0")/labenv.sh"`), which live in tools/: the checkout is found from the
# calling script's own path, so a worktree's scripts run that worktree's build and references. It changes to the checkout
# (relative paths in arguments are taken from there) and sets:
#   FNX  the checkout       M   the models' directory (models, a link)        G   the Q4 model's first shard (UD-Q4_K_XL)
#   Q8   the Q8_0 model's first shard                        P   tools/trace (the cupti tracer and its readers)
#   PY   a python with numpy for ref/*.py: .venv's, or in a worktree (no venv of its own) the main checkout's
# G, Q8 and PY can be set by the caller. Functions:
#   unit ENVS... -- COMMAND...   one GPU program as a transient unit: memlock unlimited (the engine pins host memory),
#                                this user, in fnx/; its exit status is the program's. UNIT_TIMEOUT (seconds, default
#                                2400: the longest run here, a 62k prompt read and its decode on Q8_0, takes ~15
#                                minutes) ends a run hung on a kernel waiting for a peer that never comes (status 124)
#   sampled OUT COMMAND...       COMMAND's output to OUT, the GPUs polled once a second to OUT.gpu while it runs, then
#                                one line of their mean utilization and power; its exit status is COMMAND's
#   lab_tmp                      O: a new directory for this run's files
#   run NAME ENVS... -- COMMAND...  unit with its output to $O/NAME.log; a non-zero exit is failed, with the log's tail
#   same A B                     both files exist, are not empty, and are byte for byte equal
#   failed WHAT                  says WHAT failed now and lists it again at the end
#   finish                       the script's end: exit 1 when anything failed (keeping $O for its logs), else exit
#                                0, removing $O unless KEEP_O is set (a script whose files are its output)
# A check must fail visibly: capture a program's status before filtering its output (a pipe into grep reports grep's),
# and compare only files this run wrote.
FNX=$(cd "$(dirname "$0")/.." && pwd) || exit 1
cd "$FNX" || exit 1
M=$FNX/models
G=${G:-$M/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}
Q8=${Q8:-$M/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf}
# shellcheck disable=SC2034   # used by the scripts that source this
P=$FNX/tools/trace
if [ -z "${PY:-}" ]; then
    PY=$FNX/.venv/bin/python
    if [ ! -x "$PY" ] && common=$(git -C "$FNX" rev-parse --path-format=absolute --git-common-dir 2> /dev/null); then
        PY=$(dirname "$common")/.venv/bin/python
    fi
fi
FAILED=""
O=""

need_py() {
    [ -x "$PY" ] || { echo "no python at $PY: make fnx/.venv (numpy) or set PY="; exit 1; }
}

unit() {
    # shellcheck disable=SC2086
    timeout "${UNIT_TIMEOUT:-2400}" sudo systemd-run --quiet --wait --pipe --collect -p LimitMEMLOCK=infinity \
        --uid="$(id -u)" --gid="$(id -g)" -p WorkingDirectory="$FNX" "$@"
}

sampled() {
    _s_out=$1
    shift
    nvidia-smi --query-gpu=index,utilization.gpu,power.draw --format=csv,noheader,nounits -lms 1000 > "$_s_out.gpu" &
    _s_smi=$!
    "$@" > "$_s_out" 2>&1
    _s_rc=$?
    kill $_s_smi
    _s_gpu=$(awk -F', ' '{u += $2; p += $3; n++} END {if (n) printf "util %.0f%% power %.0f W", u / n, p / n}' "$_s_out.gpu")
    if [ $_s_rc = 0 ]; then echo "== $(basename "$_s_out" .txt): $_s_gpu"; else echo "== $(basename "$_s_out" .txt): $_s_gpu; exit $_s_rc"; fi
    return $_s_rc
}

lab_tmp() {
    O=$(mktemp -d "${TMPDIR:-/tmp}/fnx-check.XXXXXX") || { echo "mktemp failed"; exit 1; }
}

failed() {
    echo "FAILED: $1"
    FAILED="$FAILED
  $1"
}

run() {
    [ -n "$O" ] || { echo "run before lab_tmp"; exit 1; }
    _r_name=$1
    shift
    unit "$@" > "$O/$_r_name.log" 2>&1
    _r_rc=$?
    if [ $_r_rc != 0 ]; then
        failed "$_r_name: exit $_r_rc (log $O/$_r_name.log)"
        tail -5 "$O/$_r_name.log"
    fi
    return $_r_rc
}

same() {
    [ -s "$1" ] && [ -s "$2" ] && cmp -s "$1" "$2"
}

finish() {
    if [ -n "$FAILED" ]; then
        echo "== FAILED:$FAILED"
        [ -n "$O" ] && echo "(this run's files: $O)"
        exit 1
    fi
    if [ -n "$O" ] && [ -n "${KEEP_O:-}" ]; then
        echo "(this run's files: $O)"
    elif [ -n "$O" ] && [ -d "$O" ]; then
        # only the directory lab_tmp made: mktemp's name under TMPDIR, never anything else O might hold
        case "$O" in
        "${TMPDIR:-/tmp}"/fnx-check.??????) rm -rf -- "$O" ;;
        *) echo "(not removing $O: not lab_tmp's)" ;;
        esac
    fi
    exit 0
}
