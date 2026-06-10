#!/bin/bash
#SBATCH --job-name=centrality-preprocessing
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=centrality-preporocessing_%j.out
#SBATCH --error=centrality-preprocessing_%j.err

echo "========================================="
echo "Job Started"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Start Time  : $(date)"
echo "========================================="

module load cuda-12.1

# Load conda
source /data/surendra/workspace/anaconda3/etc/profile.d/conda.sh
conda activate dgl-centrality

echo "Python Path : $(which python)"
echo "Conda Env   : $CONDA_DEFAULT_ENV"
echo "CUDA Visible Devices : $CUDA_VISIBLE_DEVICES"
echo ""

# Run datasets one by one
datasets=("ogbn-arxiv" "ogbn-products" "reddit" "igb-small" "yelp")

for dataset in "${datasets[@]}"
do
    echo "========================================="
    echo "Running dataset: ${dataset}"
    echo "Start Time: $(date)"
    echo "========================================="

    python dissimilarity_eigenvector_centrality.py \
        --dataset ${dataset} \
        --max_iter 100

    echo "Finished dataset: ${dataset}"
    echo "End Time: $(date)"
    echo ""
done

echo "========================================="
echo "All jobs completed"
echo "Finish Time : $(date)"
echo "========================================="
