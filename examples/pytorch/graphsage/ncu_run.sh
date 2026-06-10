#!/bin/bash

set -e

if [ $# -lt 3 ]; then
    echo "Usage:"
    echo "$0 <output_name> <kernel_regex> <application> [args...]"
    exit 1
fi

OUT=$1
KERNEL=$2
shift 2

APP=$1
shift

ncu \
    --kernel-name "$KERNEL" \
    --section SpeedOfLight \
    -o "$OUT" \
    "$APP" "$@"

echo
echo "Report saved:"
echo "${OUT}.ncu-rep"
