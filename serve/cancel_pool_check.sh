#!/bin/sh
# The cancel test on the served config (serve/sfnx.json), then the pool test on the small-context config
# (serve/sfnx-pool.json): the short-reply version, and long replies that outgrow the pool after admission (decoding
# slots paused and resumed). The test unit on port 8095, stopped at the end.
#   serve/cancel_pool_check.sh
HERE=$(cd "$(dirname "$0")" && pwd)
URL=http://127.0.0.1:8095/v1
up() {
    for _ in $(seq 120); do curl -sf -m 2 $URL/models > /dev/null && return 0; sleep 5; done
    echo "sfnx did not come up"; journalctl -u sfnx-test -n 5 --no-pager; exit 1
}
"$HERE/run.sh" "$HERE/sfnx.json" > /dev/null 2>&1 && up
echo "== cancel_test"
python3 "$HERE/cancel_test.py" $URL "$HERE/sfnx.log"
"$HERE/run.sh" "$HERE/sfnx-pool.json" > /dev/null 2>&1 && up
from=$(stat -c %s "$HERE/sfnx-pool.log")
echo "== pool_test, 6 x 9k words, short replies"
python3 "$HERE/pool_test.py" $URL
echo "== pool_test, 3 x 7k words, 4,500-token copies"
python3 "$HERE/pool_test.py" $URL 3 7000 4500 copy
echo "== the pool's events"
tail -c +$((from + 1)) "$HERE/sfnx-pool.log" | grep -E "paused|resumed|pool full|pool short" | sort | uniq -c | sort -rn | head -12
sudo systemctl stop sfnx-test
