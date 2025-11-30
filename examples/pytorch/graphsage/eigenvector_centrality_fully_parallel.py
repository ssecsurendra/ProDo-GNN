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

# CUDA kernel for sorting neighbors with a hybrid strategy
parallel_sort_neighbors_kernel = cp.RawKernel(r'''
extern "C" __global__
void parallel_sort_neighbors(
    const long* row_ptr,
    const long* col_idx,
    const float* centrality,
    long* sorted_col_idx,
    int num_nodes) {
    
    // Each block processes one node 'u'
    int u = blockIdx.x;
    if (u >= num_nodes) {
        return;
    }

    long start = row_ptr[u];
    int num_neighbors = row_ptr[u+1] - start;

    // If 0 or 1 neighbors, no sorting needed.
    if (num_neighbors <= 1) {
        if (threadIdx.x == 0) {
            if (num_neighbors == 1) {
                sorted_col_idx[start] = col_idx[start];
            }
        }
        return;
    }

    // --- HYBRID SORTING STRATEGY ---
    if (num_neighbors <= blockDim.x) {
        // --- FAST PATH: Parallel Odd-Even Sort for most nodes ---
        extern __shared__ unsigned char shmem[];
        float* keys = (float*)shmem;
        long* values = (long*)(shmem + blockDim.x * sizeof(float));

        // Parallel load into shared memory
        if (threadIdx.x < num_neighbors) {
            long neighbor_id = col_idx[start + threadIdx.x];
            keys[threadIdx.x] = centrality[neighbor_id];
            values[threadIdx.x] = neighbor_id;
        }
        __syncthreads();

        // Parallel Odd-Even Transposition Sort
        for (int p = 0; p < num_neighbors; p++) {
            int i = threadIdx.x;
            if ((p % 2) == 0) { // Even phase
                if ((i % 2) == 0 && (i + 1 < num_neighbors)) {
                    if (keys[i] < keys[i+1]) {
                        float temp_key = keys[i]; keys[i] = keys[i+1]; keys[i+1] = temp_key;
                        long temp_val = values[i]; values[i] = values[i+1]; values[i+1] = temp_val;
                    }
                }
            } else { // Odd phase
                if ((i % 2) != 0 && (i + 1 < num_neighbors)) {
                     if (keys[i] < keys[i+1]) {
                        float temp_key = keys[i]; keys[i] = keys[i+1]; keys[i+1] = temp_key;
                        long temp_val = values[i]; values[i] = values[i+1]; values[i+1] = temp_val;
                    }
                }
            }
            __syncthreads();
        }

        // Parallel write back to global memory
        if (threadIdx.x < num_neighbors) {
            sorted_col_idx[start + threadIdx.x] = values[threadIdx.x];
        }

    } else {
        // --- FALLBACK PATH: Single-threaded Bubble Sort for large degree nodes ---
        if (threadIdx.x == 0) {
            // Copy neighbors to be sorted in-place
            for (int i = 0; i < num_neighbors; ++i) {
                sorted_col_idx[start + i] = col_idx[start + i];
            }

            // Bubble sort
            for (int i = 0; i < num_neighbors - 1; ++i) {
                for (int j = 0; j < num_neighbors - i - 1; ++j) {
                    long neighbor1_id = sorted_col_idx[start + j];
                    long neighbor2_id = sorted_col_idx[start + j + 1];
                    
                    if (centrality[neighbor1_id] < centrality[neighbor2_id]) {
                        long temp = sorted_col_idx[start + j];
                        sorted_col_idx[start + j] = sorted_col_idx[start + j + 1];
                        sorted_col_idx[start + j + 1] = temp;
                    }
                }
            }
        }
    }
}
''', 'parallel_sort_neighbors')


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
    print(f"Calculating Eigenvector Centrality for {args.dataset} using FULLY PARALLEL sort kernel.")

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

    G = data[0]
    if G.device.type != 'cuda':
        G = G.to('cuda')
    
    total_start = time.time()

    eigen_centrality_cp = eigenvector_centrality_power_iteration(G, max_iter=args.max_iter)
    
    row_ptr_cp = cp.asarray(G.adj_tensors('csr')[0])
    col_idx_cp = cp.asarray(G.adj_tensors('csr')[1])
    
    num_nodes = G.num_nodes()
    sorted_col_idx_cp = cp.empty_like(col_idx_cp)

    # --- Kernel Launch ---
    threads_per_block = 1024 
    blocks = num_nodes

    # Warn if max degree exceeds block size, but run anyway with the fallback.
    max_degree = int(G.out_degrees().max().item())
    if max_degree > threads_per_block:
        print(f"Warning: Max degree of graph ({max_degree}) exceeds threads per block ({threads_per_block}).")
        print(f"         Nodes with degree > {threads_per_block} will use a slower, sequential sorting path inside the kernel.")

    # Shared memory size: (size of keys) + (size of values)
    # Keys are float (4 bytes), values are long (8 bytes)
    shared_mem_size = (threads_per_block * 4) + (threads_per_block * 8)

    print("Sorting neighbors on GPU with fully parallel kernel...")
    sorting_start_time = time.time()
    parallel_sort_neighbors_kernel(
        (blocks,), (threads_per_block,),
        shared_mem=shared_mem_size,
        args=(row_ptr_cp, col_idx_cp, eigen_centrality_cp, sorted_col_idx_cp, num_nodes)
    )
    cp.cuda.Device(0).synchronize()
    sorting_end_time = time.time()
    print(f"GPU fully parallel kernel sorting time: {sorting_end_time - sorting_start_time:.4f} seconds")

    out_dir = "eigenvector-centrality-fully-parallel"
    os.makedirs(out_dir, exist_ok=True)

    filename_centrality = os.path.join(out_dir, f"{args.dataset}_eigen-centrality.txt")
    filename_sorted_col_idx = os.path.join(out_dir, f"{args.dataset}_eigen_sorted-col-index.txt")
    
    np.savetxt(filename_centrality, eigen_centrality_cp.get(), fmt="%.10f")
    np.savetxt(filename_sorted_col_idx, sorted_col_idx_cp.get(), fmt="%d")
    
    print(f"✅ Eigenvector centrality saved to {filename_centrality}")
    print(f"✅ Sorted column-index saved to {filename_sorted_col_idx}")
    
    total_end = time.time()
    print(f"Total time: {total_end - total_start:.4f} Seconds")
