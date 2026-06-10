#!/bin/bash

if [ $# -lt 4 ]; then
    echo "Usage:"
    echo "$0 report.ncu-rep kernel_name metric_name output_prefix"
    exit 1
fi

REPORT=$1
KERNEL=$2
METRIC=$3
OUT=$4

TMPCSV=$(mktemp)

ncu \
    --import "$REPORT" \
    --csv \
    --page raw > "$TMPCSV"

HEADER=$(head -1 "$TMPCSV")

KERNEL_COL=$(echo "$HEADER" | tr ',' '\n' | nl | grep "Kernel Name" | awk '{print $1}')
METRIC_COL=$(echo "$HEADER" | tr ',' '\n' | nl | grep "$METRIC" | awk '{print $1}')

if [ -z "$METRIC_COL" ]; then
    echo "Metric not found"
    exit 1
fi

awk -F',' \
-v kernel="$KERNEL" \
-v kcol="$KERNEL_COL" \
-v mcol="$METRIC_COL" '
BEGIN{
    idx=0;
}
NR>1{
    if(index($kcol,kernel)){
        idx++;
        print idx " " $mcol;
    }
}' "$TMPCSV" > ${OUT}.dat

gnuplot << EOF

set terminal pngcairo size 1600,800
set output "${OUT}.png"

set title "${METRIC}"
set xlabel "Kernel Invocation"
set ylabel "${METRIC}"

set grid
set key off

plot "${OUT}.dat" using 1:2 with linespoints lw 2 pt 7

EOF

echo "Generated ${OUT}.png"

rm -f "$TMPCSV"
