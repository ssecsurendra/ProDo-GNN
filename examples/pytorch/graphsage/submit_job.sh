#!/bin/bash
#SBATCH --job-name=centrality
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=centrality_%j.out
#SBATCH --error=centrality_%j.err
#module load  anaconda3
#module load dgl-2.5.0-cu121
#module load python-3.10.13
module load cuda-12.1
# Load conda
source /data/surendra/workspace/anaconda3/etc/profile.d/conda.sh
conda activate dgl-centrality

#echo "After activation:"
#which python
#echo $CONDA_DEFAULT_ENV
./run_script_surendra_TACO.sh ogbn-arxiv 100 
./run_script_surendra_TACO.sh ogbn-products 100 
./run_script_surendra_TACO.sh igb-small 100 
./run_script_surendra_TACO.sh reddit 100 
#./run_script_surendra_TACO.sh yelp 100 

# ./run_script_surendra_TACO.sh ogbn-arxiv 100 

