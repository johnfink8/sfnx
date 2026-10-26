#!/bin/sh
# fnx_run's mtp mode on one prompt with the memlock limit raised: prompt, tokens to generate, then optional output
# file (paths from fnx/). Env: CTX (default 131072), K (draft chain, default 4), LK (lookup drafts, default 7),
# ENVS="-E VAR=value ...". Its exit status is fnx_run's.
#   tools/mtp_run.sh ref/long114.bin 512 [OUT.txt]
. "$(dirname "$0")/labenv.sh"
# shellcheck disable=SC2086
unit -E FNX_MAX_CTX="${CTX:-131072}" ${ENVS:-} -- bin/fnx_run "$G" mtp "$FNX/data/mtp-rt" "$1" "$2" "${K:-4}" \
    "$FNX/data/draft_vocab_en.bin" "${LK:-7}" > "${3:-/dev/stdout}" 2>&1
