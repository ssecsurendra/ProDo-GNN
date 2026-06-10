#!/bin/bash

if [ $# -lt 4 ]; then
    echo "Usage:"
    echo "$0 report.ncu-rep kernel_name metric output_prefix"
    exit 1
fi

REPORT=$1
KERNEL=$2
METRIC=$3
OUT=$4

CSV=$(mktemp)

echo "[1/3] Exporting report..."

ncu --import "$REPORT" \
    --csv \
    --page raw > "$CSV"

echo "[2/3] Extracting metric..."

python3 << EOF

import pandas as pd

df = pd.read_csv("$CSV")

kernel_col = "Kernel Name"

df = df[df[kernel_col].astype(str).str.contains("$KERNEL", regex=False)]

if len(df) == 0:
    print("Kernel not found")
    exit(1)

metric = "$METRIC"

if metric not in df.columns:
    print("Metric not found")
    print("\nAvailable throughput/cycle metrics:\n")

    for c in df.columns:
        if "throughput" in c.lower() or "cycle" in c.lower():
            print(c)

    exit(1)

df = df.reset_index(drop=True)

with open("${OUT}.dat","w") as f:
    for idx,val in enumerate(df[metric]):
        f.write(f"{idx} {val}\n")

print("Kernel launches:", len(df))

EOF

echo "[3/3] Generating gnuplot..."

gnuplot << EOF

set terminal pngcairo size 1800,800
set output "${OUT}.png"

set title "${KERNEL}\n${METRIC}"

set xlabel "Kernel Invocation"
set ylabel "${METRIC}"

set grid
set key off

# Batch boundary every 5 launches
do for [i=5:10000:5] {
    set arrow from i,graph 0 to i,graph 1 nohead dt 2
}

plot "${OUT}.dat" using 1:2 \
     with linespoints lw 2 pt 7

EOF

rm -f "$CSV"

echo
echo "Generated: ${OUT}.png"
