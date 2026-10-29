#!/bin/sh
# The Makefile's checks of the build settings, on the CPU: every bad ARCH and FNX_NCARD stops make with its own message
# before a stamp is written, and a value padded with spaces is the same setting (no stamp rewritten, so nothing
# rebuilds). Run in a checkout that has been built (build/arch and build/ncard exist). Exits 1 after reporting every
# difference.
#   tools/build_guards_check.sh
cd "$(dirname "$0")/.." || exit 1
[ -f build/arch ] && [ -f build/ncard ] || { echo "build_guards_check: no build/arch or build/ncard: build first (make)"; exit 1; }
fails=0
failed() {
    echo "FAIL: $1"
    fails=$((fails + 1))
}
# WANT ARGS...: make -n with ARGS stops, saying WANT
refused() {
    want=$1
    shift
    if out=$(make -n "$@" build/arch 2>&1); then
        failed "make $* passed"
    else
        case "$out" in
            *"$want"*) ;;
            *) failed "make $* stopped with '$out', want '$want'" ;;
        esac
    fi
}
arch=$(cat build/arch) ncard=$(cat build/ncard)
mark=$(mktemp) || exit 1
refused "ARCH: 75 (Turing)" ARCH=75
refused "ARCH: one value, not '86 90'" "ARCH=86 90"
refused "not '81'" ARCH=81
refused "not 'sm_86'" ARCH=sm_86
refused "FNX_NCARD: one value, not '3 4'" "FNX_NCARD=3 4"
refused "not '5'" FNX_NCARD=5
# the same settings padded with spaces, from the command line and from the environment
make -n ARCH=" $arch " FNX_NCARD=" $ncard" build/arch > /dev/null 2>&1 || failed "make with ARCH=' $arch ' FNX_NCARD=' $ncard' stopped"
ARCH=" $arch" FNX_NCARD="$ncard " make -n build/arch > /dev/null 2>&1 || failed "make with ARCH, FNX_NCARD padded in the environment stopped"
for s in build/arch build/ncard; do
    [ -n "$(find "$s" -newer "$mark")" ] && failed "$s was rewritten by a refused or padded make"
done
[ "$(cat build/arch)" = "$arch" ] && [ "$(cat build/ncard)" = "$ncard" ] || failed "the stamps changed: $(cat build/arch) $(cat build/ncard)"
unlink "$mark"
if [ $fails -gt 0 ]; then
    echo "build_guards_check: $fails failed"
    exit 1
fi
echo "build_guards_check: ok"
