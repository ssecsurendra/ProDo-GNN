import numpy as np
import sys
import dgl
import time
import os
import torch as th
import cupy as cp
import argparse
from cupy.sparse import csr_matrix
os.environ["DGLBACKEND"] = "pytorch"
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, WisconsinDataset, FlickrDataset, RedditDataset, YelpDataset

# CUDA kernel for sorting neighbors based on centrality
sort_neighbors_kernel = cp.RawKernel(r'''
extern "C" __global__
void sort_neighbors(
    const long* row_ptr,
    const long* col_idx,
    const float* centrality,
    long* sorted_col_idx,
    int num_nodes) {

    int u = blockIdx.x;

    if (u >= num_nodes) {
        return;
    }

    // This thread (thread 0 in the block) will sort the neighbors for node u.
    if (threadIdx.x == 0) {
        long start = row_ptr[u];
        long end = row_ptr[u+1];
        int num_neighbors = end - start;

        if (num_neighbors == 0) {
            return;
        }

        // Copy neighbors to the output array to be sorted in-place.
        for (int i = 0; i < num_neighbors; ++i) {
            sorted_col_idx[start + i] = col_idx[start + i];
        }

        // Simple bubble sort on the slice of sorted_col_idx.
        // This is inefficient for large lists but parallel across nodes,
        // avoiding the Python loop overhead.
        for (int i = 0; i < num_neighbors - 1; ++i) {
            for (int j = 0; j < num_neighbors - i - 1; ++j) {
                long neighbor1_node_id = sorted_col_idx[start + j];
                long neighbor2_node_id = sorted_col_idx[start + j + 1];
                
                if (centrality[neighbor1_node_id] < centrality[neighbor2_node_id]) {
                    // Swap
                    long temp = sorted_col_idx[start + j];
                    sorted_col_idx[start + j] = sorted_col_idx[start + j + 1];
                    sorted_col_idx[start + j + 1] = temp;
                }
            }
        }
    }
}
''', 'sort_neighbors')


def eigenvector_centrality_power_iteration(g, max_iter=100, tol=1e-6):
    """
    Calculates eigenvector centrality using the power iteration method on a GPU.
    """
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
        if norm == 0:
            return x
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-arxiv",
        help="Dataset name ('cora', 'flickr', 'reddit', 'ogbn-products', etc.)"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum iterations for power iteration."
    )
    args = parser.parse_args()
    print(f"Calculating Eigenvector Centrality for {args.dataset} using GPU with parallel sort kernel.")

    # Load and preprocess dataset
    try:
        if args.dataset == "cora":
            data = CoraGraphDataset()
        elif args.dataset == "citeseer":
            data = CiteseerGraphDataset()
        elif args.dataset == "pubmed":
            data = PubmedGraphDataset()
        elif args.dataset == "wisconsin":
            data = WisconsinDataset()
        elif args.dataset == "flickr":
            data = FlickrDataset()
        elif args.dataset == "reddit":
            data = RedditDataset()
        elif args.dataset == "yelp":
            data = YelpDataset()
        elif args.dataset == "ogbn-products":
            data = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
        elif args.dataset == "ogbn-arxiv":
            data = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv"))
        elif args.dataset == "ogbn-papers":
            data = AsNodePredDataset(DglNodePropPredDataset("ogbn-papers100M"))
        elif args.dataset == "igb-small":
            load_path = './dataset/igb_small.dgl'
            data, _ = dgl.load_graphs(load_path)
        else:
            raise ValueError("Unknown dataset: {}".format(args.dataset))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    G = data[0]
    
    # Per user request, assume graph is undirected and do not convert.
    
    total_start = time.time()

    # Calculate Eigenvector Centrality using GPU
    eigen_centrality_cp = eigenvector_centrality_power_iteration(G, max_iter=args.max_iter)
    
    # --- Sort column indices based on centrality using the custom CUDA kernel ---
    row_ptr_cp = cp.asarray(G.adj_tensors('csr')[0])
    col_idx_cp = cp.asarray(G.adj_tensors('csr')[1])
    
    num_nodes = G.num_nodes()
    sorted_col_idx_cp = cp.empty_like(col_idx_cp)

    # Kernel launch configuration
    threads_per_block = 128  # Only thread 0 does work, but this is a standard size.
    blocks = num_nodes      # One block per node

    print("Sorting neighbors on GPU with custom kernel...")
    sorting_start_time = time.time()
    sort_neighbors_kernel(
        (blocks,), (threads_per_block,),
        (row_ptr_cp, col_idx_cp, eigen_centrality_cp, sorted_col_idx_cp, num_nodes)
    )
    cp.cuda.Device(0).synchronize()
    sorting_end_time = time.time()
    print(f"GPU kernel sorting time: {sorting_end_time - sorting_start_time:.4f} seconds")


    # --- ensure output folder exists ---
    out_dir = "eigenvector-centrality-gpu-kernel"
    os.makedirs(out_dir, exist_ok=True)

    # --- save results ---
    filename_centrality = os.path.join(out_dir, f"{args.dataset}_eigen-centrality.txt")
    filename_sorted_col_idx = os.path.join(out_dir, f"{args.dataset}_eigen_sorted-col-index.txt")
    
    # Transfer data back to CPU for saving
    np.savetxt(filename_centrality, eigen_centrality_cp.get(), fmt="%.10f")
    np.savetxt(filename_sorted_col_idx, sorted_col_idx_cp.get(), fmt="%d")
    
    print(f"✅ Eigenvector centrality saved to {filename_centrality}")
    print(f"✅ Sorted column-index saved to {filename_sorted_col_idx}")
    
    total_end = time.time()
    print(f"Total time: {total_end - total_start:.4f} Seconds")
