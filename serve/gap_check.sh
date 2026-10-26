#!/bin/sh
# serve/gap_test.py on the arena with a tiny own pool (440 pages: the six lanes' blocks and 30 pages of K/V), once per
# set of engine variables, each on a fresh start of the test unit (port 8095); prints whether the engine failed.
#   serve/gap_check.sh serve/sfnx-gap.json "LABEL=-E VAR=value ..." ...      ("plain=" for no variables)
HERE=$(cd "$(dirname "$0")" && pwd)
CFG=$1
shift
LOG=$(sed -n 's/.*"log": "\([^"]*\)".*/\1/p' "$CFG")
for spec in "$@"; do
    label=${spec%%=*} envs=${spec#*=}
    ENVS="$envs" "$HERE/run.sh" "$CFG" > /dev/null 2>&1
    for _ in $(seq 100); do
        curl -sf -m 2 http://127.0.0.1:8095/v1/models > /dev/null && break
        [ "$(systemctl is-active sfnx-test)" = failed ] && break
        sleep 4
    done
    from=$(stat -c %s "$LOG")
    echo "== $label ($envs)"
    timeout 1200 python3 "$HERE/gap_test.py" http://127.0.0.1:8095/v1 4000 2>&1 | tail -3 | cut -c1-160
    tail -c +$((from + 1)) "$LOG" | grep -E "failed" | head -2
    echo "engine failures: $(tail -c +$((from + 1)) "$LOG" | grep -c 'sfnx: failed'); arena changes: $(tail -c +$((from + 1)) "$LOG" | grep -c 'arena: K/V pool')"
    sudo systemctl stop sfnx-test
done
