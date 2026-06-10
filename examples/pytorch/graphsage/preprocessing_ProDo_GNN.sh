#!/bin/bash

# Activate conda environment
#source /data/surendra/workspace/anaconda3/etc/profile.d/conda.sh
#conda activate dgl-centrality

DATASETS=("ogbn-arxiv" "ogbn-products" "reddit" "yelp" "igb-small")

echo "========================================="
echo "Started at: $(date)"
echo "========================================="

for DATASET in "${DATASETS[@]}"
do
    echo ""
    echo "========================================="
    echo "Running dataset: ${DATASET}"
    echo "Start Time: $(date)"
    echo "========================================="

    python3 dissimilarity_eigenvector_centrality.py \
        --dataset ${DATASET} \
        --max_iter 100

    echo "Finished dataset: ${DATASET}"
    echo "End Time: $(date)"
done

echo ""
echo "========================================="
echo "All datasets completed"
echo "Finish Time: $(date)"
echo "========================================="
