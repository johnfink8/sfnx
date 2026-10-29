#!/bin/sh
# runs a program of this checkout with the memlock limit the loader needs (the embedding table locked in RAM), and the
# engine's settings (FNX_*, S27_*, CUDA_*) of this shell, which systemd-run does not pass on its own. The settings it
# passes are named on stderr, so one left exported from an earlier run shows beside the result it moved
for n in $(env | sed -n 's/^\(\(FNX\|S27\|CUDA\)_[A-Za-z0-9_]*\)=.*/\1/p'); do
    eval "v=\${$n}"
    set -- "--setenv=$n=$v" "$@"
    echo "s27.sh: $n=$v" >&2
done
exec sudo -n systemd-run --quiet --pipe --wait --collect -p LimitMEMLOCK=infinity --uid="$(id -u)" --gid="$(id -g)" \
    -p WorkingDirectory="$(pwd)" "$@"
