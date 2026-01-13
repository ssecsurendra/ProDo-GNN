
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
import dgl.function as fn
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
        
        edge_weights[e] = 1.0f-fmaxf(raw_cosine, 0.0f);
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
    selection_sort_kernel((blocks_per_grid,), (threads_per_block,), (indptr, indices, centrality, num_rows))
    return indices

def weighted_eigenvector_centrality(g, edge_weights_cp, max_iter=100, tol=1e-6):
    if g.device.type != 'cuda':
        g = g.to('cuda')
    indptr, indices, _ = g.adj_tensors('csr')
    indptr_cp = cp.asarray(indptr)
    indices_cp = cp.asarray(indices)
    num_nodes = g.num_nodes()
    
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

def compute_and_save_dominating_set(g, centrality_values_cp, dataset_name, out_dir):
    print("\n--- Dominating Set Calculation ---")
    dom_set_start_time = time.time()

    # 1. Sort nodes by the provided centrality
    print("Sorting nodes by centrality for dominating set...")
    sorted_nodes_cp = cp.argsort(-centrality_values_cp)
    
    # Convert to PyTorch tensor for DGL operations
    device = g.device
    num_nodes = g.num_nodes()
    sorted_nodes = th.from_dlpack(sorted_nodes_cp.toDlpack()).to(device)

    # 2. Create position map
    print("Creating position map for dominating set...")
    large_float = float(num_nodes)
    pos_map = th.full((num_nodes,), large_float, dtype=th.float32, device=device)
    pos_map[sorted_nodes] = th.arange(num_nodes, device=device, dtype=th.float32)
    g.ndata['min_rank'] = pos_map

    # 3. Message Passing (1-hop)
    print("Starting message passing for dominating set...")
    g.update_all(fn.copy_u('min_rank', 'm'), fn.min('m', 'neighbor_min_rank'))
    final_min_pos = th.min(g.ndata['min_rank'], g.ndata['neighbor_min_rank']).long()

    # 4. Get the unique dominating nodes
    dominating_nodes = sorted_nodes[final_min_pos]
    dominating_set = th.unique(dominating_nodes)
    
    dom_set_end_time = time.time()
    computation_time = dom_set_end_time - dom_set_start_time
    
    # 5. Save the result
    filename_dom_set = os.path.join(out_dir, f"{dataset_name}_eigen_dominating_set.npy")
    np.save(filename_dom_set, dominating_set.cpu().numpy())
    
    print(f"✅ Dominating set saved to {filename_dom_set}")
    print(f"Number of nodes in dominating set: {len(dominating_set)}")

    # Return the time for the final summary
    return computation_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv",
                        choices=['reddit', 'ogbn-products', 'ogbn-arxiv', 'yelp', 'igb-small', 'cora', 'citeseer', 'pubmed'])
    parser.add_argument("--max_iter", type=int, default=100)
    args = parser.parse_args()
    
    print(f"Processing dataset: {args.dataset}")

    # Load and preprocess dataset
    try:
        if args.dataset == "cora": data = CoraGraphDataset()
        elif args.dataset == "citeseer": data = CiteseerGraphDataset()
        elif args.dataset == "pubmed": data = PubmedGraphDataset()
        elif args.dataset == "reddit": data = RedditDataset()
        elif args.dataset == "ogbn-products": data = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
        elif args.dataset == "ogbn-arxiv": data = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv"))
        elif args.dataset == "yelp": data = YelpDataset()
        elif args.dataset == "igb-small": 
            load_path = './dataset/igb_small.dgl' 
            data, _ = dgl.load_graphs(load_path)
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
    print("\n--- Edge Weight Calculation ---")
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
    print("\n--- Weighted Eigenvector Centrality Calculation ---")
    eigen_centrality_cp = weighted_eigenvector_centrality(G, edge_weights_cp, max_iter=args.max_iter)
    cp.cuda.Device(0).synchronize()
    eigen_end_time = time.time()

    # --- 3. Sorting (for original script's purpose) ---
    sorted_col_idx_cp = col_idx_cp.copy()
    sorting_start_time = time.time()
    print("\n--- Sorting Neighbors by Centrality ---")
    gpu_sort_csr_by_centrality(row_ptr_cp, sorted_col_idx_cp, eigen_centrality_cp)
    cp.cuda.Device(0).synchronize()
    sorting_end_time = time.time()

    # --- 4. Dominating Set Calculation (New) ---
    dom_set_time = compute_and_save_dominating_set(G, eigen_centrality_cp, args.dataset, "dissimilarity-eigenvector-centrality")

    # --- 5. Save original results ---
    out_dir = "dissimilarity-eigenvector-centrality"
    os.makedirs(out_dir, exist_ok=True)
    filename_centrality = os.path.join(out_dir, f"{args.dataset}_dissimilar_eigen-centrality.npy")
    filename_sorted_col_idx = os.path.join(out_dir, f"{args.dataset}_dissimilar_eigen_sorted-col-index.npy")
    
    save_start_time = time.time()
    np.save(filename_centrality, eigen_centrality_cp.get())
    np.save(filename_sorted_col_idx, sorted_col_idx_cp.get())
    save_end_time = time.time()
    
    overall_end_time = time.time()

    # --- 6. Print Timings ---
    print("\n--- Execution Timings ---")
    print(f"Weight Calculation Time     : {weight_calc_end_time - weight_calc_start_time:.4f} seconds")
    print(f"Eigenvector Calculation Time: {eigen_end_time - eigen_start_time:.4f} seconds")
    print(f"Neighbor Sorting Time       : {sorting_end_time - sorting_start_time:.4f} seconds")
    print(f"Dominating Set Calc Time    : {dom_set_time:.4f} seconds")
    print(f"File Save Time (binary)     : {save_end_time - save_start_time:.4f} seconds")
    print(f"Overall Execution Time      : {overall_end_time - overall_start_time:.4f} seconds (post-graph-loading)")
    print("-------------------------\n")
    
    print(f"✅ Weighted eigenvector centrality saved to {filename_centrality}")
    print(f"✅ Sorted column-index saved to {filename_sorted_col_idx}")
