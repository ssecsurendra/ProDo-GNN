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

# Kernel to calculate cosine similarity for each edge
similarity_kernel = cp.RawKernel(r'''
extern "C" __global__
void calculate_cosine_similarity(
    const float* features, 
    const long* indptr, 
    const long* indices, 
    float* edge_weights, 
    int feature_dim,
    int num_nodes
) {
    int src = blockIdx.x;
    if (src >= num_nodes) return;

    long edge_start = indptr[src];
    long edge_end = indptr[src + 1];

    for (long e = edge_start + threadIdx.x; e < edge_end; e += blockDim.x) {
        long dst = indices[e];

        float dot = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;

        const float* feature_a = features + (long)src * feature_dim;
        const float* feature_b = features + (long)dst * feature_dim;

        for (int i = 0; i < feature_dim; ++i) {
            float a = feature_a[i];
            float b = feature_b[i];
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        float raw_cosine = 0.0f;
        if (norm_a > 0.0f && norm_b > 0.0f) {
            raw_cosine = dot / (sqrtf(norm_a) * sqrtf(norm_b));
        }
        
        // Clamp to minimum 0.0 as negative weights can make eigenvector centrality problematic
        edge_weights[e] = fmaxf(raw_cosine, 0.0f);
    }
}
''', 'calculate_cosine_similarity')

# Kernel for sorting neighbors using Selection Sort
selection_sort_kernel = cp.RawKernel(r'''
extern "C" __global__
void csr_sort_by_centrality(const long* indptr, long* indices,
                            const float* centrality, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    long start = indptr[row], end = indptr[row + 1];
    int len = end - start;
    if (len <= 1) return;
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
        if (max_idx != i) {
            long temp = indices[start + i];
            indices[start + i] = indices[start + max_idx];
            indices[start + max_idx] = temp;
        }
    }
}
''', 'csr_sort_by_centrality')

def gpu_sort_csr_by_centrality(indptr, indices, centrality):
    num_rows = indptr.size - 1
    threads_per_block = 128
    blocks_per_grid = (num_rows + threads_per_block - 1) // threads_per_block
    selection_sort_kernel((blocks_per_grid,), (threads_per_block,),
                          (indptr, indices, centrality, num_rows))
    return indices

def weighted_eigenvector_centrality(g, edge_weights_cp, max_iter=100, tol=1e-6):
    if g.device.type != 'cuda':
        g = g.to('cuda')
    indptr, indices, _ = g.adj_tensors('csr')
    indptr_cp = cp.asarray(indptr)
    indices_cp = cp.asarray(indices)
    num_nodes = g.num_nodes()
    
    # Use the pre-computed edge weights instead of ones
    adj_matrix = csr_matrix((edge_weights_cp, indices_cp, indptr_cp), shape=(num_nodes, num_nodes))
    
    x = cp.ones(num_nodes, dtype=cp.float32)
    for i in range(max_iter):
        x_old = cp.copy(x)
        x = adj_matrix.dot(x)
        norm = cp.linalg.norm(x)
        if norm == 0: return x
        x /= norm
        if cp.linalg.norm(x - x_old) < tol:
            print(f"Converged after {i+1} iterations.")
            break
    else:
        print(f"Did not converge within {max_iter} iterations.")
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--max_iter", type=int, default=100)
    args = parser.parse_args()
    
    print(f"Calculating WEIGHTED Eigenvector Centrality for {args.dataset}.")

    # Load and preprocess dataset
    try:
        if args.dataset == "cora": data = CoraGraphDataset()
        elif args.dataset == "citeseer": data = CiteseerGraphDataset()
        elif args.dataset == "pubmed": data = PubmedGraphDataset()
        elif args.dataset == "reddit": data = RedditDataset()
        elif args.dataset == "ogbn-products": data = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
        elif args.dataset == "ogbn-arxiv": data = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv"))
        else: raise ValueError(f"Unknown dataset: {args.dataset}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    overall_start_time = time.time()
    G = data[0]
    if G.device.type != 'cuda':
        G = G.to('cuda')
    
    # --- 1. Edge Weight Calculation ---
    weight_calc_start_time = time.time()
    print("Calculating edge weights using cosine similarity...")
    features_cp = cp.asarray(G.ndata['feat'])
    row_ptr_cp = cp.asarray(G.adj_tensors('csr')[0], dtype=cp.int64)
    col_idx_cp = cp.asarray(G.adj_tensors('csr')[1], dtype=cp.int64)
    num_nodes, feature_dim = features_cp.shape
    num_edges = col_idx_cp.size
    edge_weights_cp = cp.empty(num_edges, dtype=cp.float32)

    threads_per_block = 128
    blocks_per_grid = num_nodes
    similarity_kernel((blocks_per_grid,), (threads_per_block,), (features_cp, row_ptr_cp, col_idx_cp, edge_weights_cp, feature_dim, num_nodes))
    cp.cuda.Device(0).synchronize()
    weight_calc_end_time = time.time()

    # --- 2. Weighted Eigenvector Value Calculation ---
    eigen_start_time = time.time()
    eigen_centrality_cp = weighted_eigenvector_centrality(G, edge_weights_cp, max_iter=args.max_iter)
    cp.cuda.Device(0).synchronize()
    eigen_end_time = time.time()

    # --- 3. Sorting ---
    sorted_col_idx_cp = col_idx_cp.copy()
    sorting_start_time = time.time()
    gpu_sort_csr_by_centrality(row_ptr_cp, sorted_col_idx_cp, eigen_centrality_cp)
    cp.cuda.Device(0).synchronize()
    sorting_end_time = time.time()

    # --- 4. Save results ---
    out_dir = "similarity-eigenvector-centrality"
    os.makedirs(out_dir, exist_ok=True)
    filename_centrality = os.path.join(out_dir, f"{args.dataset}_weighted_eigen-centrality.npy")
    filename_sorted_col_idx = os.path.join(out_dir, f"{args.dataset}_weighted_eigen_sorted-col-index.npy")
    
    save_start_time = time.time()
    np.save(filename_centrality, eigen_centrality_cp.get())
    np.save(filename_sorted_col_idx, sorted_col_idx_cp.get())
    save_end_time = time.time()
    
    overall_end_time = time.time()

    # --- 5. Print Timings ---
    print("\n--- Execution Timings ---")
    print(f"Weight Calculation Time     : {weight_calc_end_time - weight_calc_start_time:.4f} seconds")
    print(f"Eigenvector Calculation Time: {eigen_end_time - eigen_start_time:.4f} seconds")
    print(f"Sorting Time                : {sorting_end_time - sorting_start_time:.4f} seconds")
    print(f"File Save Time (binary)     : {save_end_time - save_start_time:.4f} seconds")
    print(f"Overall Execution Time      : {overall_end_time - overall_start_time:.4f} seconds (post-graph-loading)")
    print("-------------------------\\n")
    
    print(f"✅ Weighted eigenvector centrality saved to {filename_centrality}")
    print(f"✅ Sorted column-index saved to {filename_sorted_col_idx}")
