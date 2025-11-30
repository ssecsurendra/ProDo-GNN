import numpy as np
import sys
import dgl
import time
import os
import torch as th
import cupy as cp
import argparse
import pandas as pd
from cupy.sparse import csr_matrix
os.environ["DGLBACKEND"] = "pytorch"
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, WisconsinDataset, FlickrDataset, RedditDataset, YelpDataset

# CUDA kernel for sorting neighbors using Selection Sort, adapted from user's reference
# Sorts based on a float 'centrality' array and uses long for graph indices.
selection_sort_kernel = cp.RawKernel(r'''
extern "C" __global__
void csr_sort_by_centrality(const long* indptr, long* indices,
                            const float* centrality, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    long start = indptr[row];
    long end = indptr[row + 1];
    int len = end - start;

    if (len <= 1) return;

    // Simple selection sort on GPU per row
    for (int i = 0; i < len - 1; ++i) {
        int max_idx = i;
        float max_centrality = centrality[indices[start + i]];
        for (int j = i + 1; j < len; ++j) {
            float cur_centrality = centrality[indices[start + j]];
            if (cur_centrality > max_centrality) {
                max_centrality = cur_centrality;
                max_idx = j;
            }
        }
        // Swap
        if (max_idx != i) {
            long temp = indices[start + i];
            indices[start + i] = indices[start + max_idx];
            indices[start + max_idx] = temp;
        }
    }
}
''', 'csr_sort_by_centrality')

def gpu_sort_csr_by_centrality(indptr, indices, centrality):
    """
    Launches the selection sort kernel.
    Note: This function modifies the 'indices' array in-place.
    """
    num_rows = indptr.size - 1
    threads_per_block = 128
    blocks_per_grid = (num_rows + threads_per_block - 1) // threads_per_block

    selection_sort_kernel((blocks_per_grid,), (threads_per_block,),
                          (indptr, indices, centrality, num_rows))
    return indices

def eigenvector_centrality_power_iteration(g, max_iter=100, tol=1e-6):
    if g.device.type != 'cuda':
        g = g.to('cuda')
    indptr, indices, _ = g.adj_tensors('csr')
    indptr_cp = cp.asarray(indptr)
    indices_cp = cp.asarray(indices)
    num_nodes = g.num_nodes()
    data_cp = cp.ones(len(indices_cp), dtype=cp.float32)
    adj_matrix = csr_matrix((data_cp, indices_cp, indptr_cp), shape=(num_nodes, num_nodes))
    x = cp.ones(num_nodes, dtype=cp.float32)
    x_old = cp.copy(x)
    for i in range(max_iter):
        x = adj_matrix.dot(x)
        norm = cp.linalg.norm(x)
        if norm == 0: return x
        x /= norm
        if cp.linalg.norm(x - x_old) < tol:
            print(f"Converged after {i+1} iterations.")
            break
        x_old = cp.copy(x)
    else:
        print(f"Did not converge within {max_iter} iterations.")
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--max_iter", type=int, default=100)
    args = parser.parse_args()
    
    print(f"Calculating Eigenvector Centrality for {args.dataset} using Selection Sort kernel.")

    # Load and preprocess dataset
    try:
        if args.dataset == "cora": data = CoraGraphDataset()
        elif args.dataset == "citeseer": data = CiteseerGraphDataset()
        elif args.dataset == "pubmed": data = PubmedGraphDataset()
        elif args.dataset == "wisconsin": data = WisconsinDataset()
        elif args.dataset == "flickr": data = FlickrDataset()
        elif args.dataset == "reddit": data = RedditDataset()
        elif args.dataset == "yelp": data = YelpDataset()
        elif args.dataset == "ogbn-products": data = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
        elif args.dataset == "ogbn-arxiv": data = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv"))
        elif args.dataset == "ogbn-papers": data = AsNodePredDataset(DglNodePropPredDataset("ogbn-papers100M"))
        elif args.dataset == "igb-small": data, _ = dgl.load_graphs('./dataset/igb_small.dgl')
        else: raise ValueError(f"Unknown dataset: {args.dataset}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    overall_start_time = time.time()

    G = data[0]
    if G.device.type != 'cuda':
        G = G.to('cuda')
    
    # --- 1. Eigenvector Value Calculation ---
    eigen_start_time = time.time()
    eigen_centrality_cp = eigenvector_centrality_power_iteration(G, max_iter=args.max_iter)
    cp.cuda.Device(0).synchronize()
    eigen_end_time = time.time()

    # --- 2. Sorting ---
    row_ptr_cp = cp.asarray(G.adj_tensors('csr')[0], dtype=cp.int64)
    col_idx_cp = cp.asarray(G.adj_tensors('csr')[1], dtype=cp.int64)
    
    # The kernel sorts in-place, so we must pass a copy.
    sorted_col_idx_cp = col_idx_cp.copy()

    sorting_start_time = time.time()
    gpu_sort_csr_by_centrality(row_ptr_cp, sorted_col_idx_cp, eigen_centrality_cp)
    cp.cuda.Device(0).synchronize()
    sorting_end_time = time.time()

    # --- Save results ---
    out_dir = "eigenvector-centrality-selection-sort"
    os.makedirs(out_dir, exist_ok=True)
    filename_centrality = os.path.join(out_dir, f"{args.dataset}_eigen-centrality.txt")
    filename_sorted_col_idx = os.path.join(out_dir, f"{args.dataset}_eigen_sorted-col-index.txt")
    
    save_start_time = time.time()
    print("Saving files using pandas.to_csv for better performance...")
    pd.DataFrame(eigen_centrality_cp.get()).to_csv(filename_centrality, header=False, index=False, float_format='%.10f')
    pd.DataFrame(sorted_col_idx_cp.get()).to_csv(filename_sorted_col_idx, header=False, index=False)
    save_end_time = time.time()
    
    overall_end_time = time.time()

    # --- 3. Print Timings ---
    print("\n--- Execution Timings ---")
    print(f"Eigenvector Calculation Time: {eigen_end_time - eigen_start_time:.4f} seconds")
    print(f"Sorting Time                : {sorting_end_time - sorting_start_time:.4f} seconds")
    print(f"File Save Time              : {save_end_time - save_start_time:.4f} seconds")
    print(f"Overall Execution Time      : {overall_end_time - overall_start_time:.4f} seconds (post-graph-loading)")
    print("-------------------------\n")
    
    print(f"✅ Eigenvector centrality saved to {filename_centrality}")
    print(f"✅ Sorted column-index saved to {filename_sorted_col_idx}")
