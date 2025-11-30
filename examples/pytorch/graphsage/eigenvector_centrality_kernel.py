import numpy as np
import sys
import dgl
import time
import os
import torch as th
import cupy as cp
import argparse
os.environ["DGLBACKEND"] = "pytorch"
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, WisconsinDataset, FlickrDataset, RedditDataset, YelpDataset

# Custom CUDA kernel for Sparse Matrix-Vector Multiplication (SpMV) for CSR format.
# This kernel calculates y = A * x, where A is a sparse matrix.
spmv_kernel = cp.RawKernel(r'''
extern "C" __global__
void spmv_csr(const int* indptr, const int* indices, const float* data,
              const float* x, float* y, int num_nodes) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < num_nodes) {
        float dot = 0.0f;
        int row_start = indptr[row];
        int row_end = indptr[row + 1];
        for (int i = row_start; i < row_end; ++i) {
            dot += data[i] * x[indices[i]];
        }
        y[row] = dot;
    }
}
''', 'spmv_csr')

def eigenvector_centrality_kernel(g, max_iter=100, tol=1e-6):
    """
    Calculates eigenvector centrality using a custom CUDA kernel for SpMV.
    """
    if g.device.type != 'cuda':
        g = g.to('cuda')

    indptr, indices, _ = g.adj_tensors('csr')
    
    indptr_cp = cp.asarray(indptr.long())
    indices_cp = cp.asarray(indices.long())
    
    num_nodes = g.num_nodes()
    data_cp = cp.ones(len(indices_cp), dtype=cp.float32)

    x = cp.ones(num_nodes, dtype=cp.float32)
    y = cp.zeros_like(x) # To store the result of SpMV

    # Kernel launch configuration
    threads_per_block = 256
    blocks_per_grid = (num_nodes + threads_per_block - 1) // threads_per_block

    for i in range(max_iter):
        x_old = x.copy()

        # Launch the custom SpMV kernel
        spmv_kernel((blocks_per_grid,), (threads_per_block,),
                    (indptr_cp, indices_cp, data_cp, x, y, num_nodes))
        
        x = y.copy() # Update x with the result

        norm = cp.linalg.norm(x)
        if norm == 0:
            return x
        x /= norm

        # Check for convergence by comparing the vectors
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
    print(f"Calculating Eigenvector Centrality for {args.dataset} using custom CUDA Kernel.")

    # Load dataset
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
    G = dgl.to_bidirected(G, copy_ndata=True)
    
    total_start = time.time()

    eigen_centrality_cp = eigenvector_centrality_kernel(G, max_iter=args.max_iter)
    
    row_ptr_cp = cp.asarray(G.adj_tensors('csr')[0].long())
    col_idx_cp = cp.asarray(G.adj_tensors('csr')[1].long())
    
    num_nodes = G.num_nodes()
    sorted_col_idx_cp = cp.empty_like(col_idx_cp)

    for u in range(num_nodes):
        start, end = row_ptr_cp[u], row_ptr_cp[u+1]
        neighbors = col_idx_cp[start:end]

        if neighbors.size > 0:
            order = cp.argsort(-eigen_centrality_cp[neighbors])
            sorted_col_idx_cp[start:end] = neighbors[order]

    out_dir = "eigenvector-centrality-kernel"
    os.makedirs(out_dir, exist_ok=True)

    filename_centrality = os.path.join(out_dir, f"{args.dataset}_eigen-centrality.txt")
    filename_sorted_col_idx = os.path.join(out_dir, f"{args.dataset}_sorted-col-index.txt")
    
    np.savetxt(filename_centrality, eigen_centrality_cp.get(), fmt="%.10f")
    np.savetxt(filename_sorted_col_idx, sorted_col_idx_cp.get(), fmt="%d")
    
    print(f"✅ Eigenvector centrality saved to {filename_centrality}")
    print(f"✅ Sorted column-index saved to {filename_sorted_col_idx}")
    
    total_end = time.time()
    print(f"Total time: {total_end - total_start:.4f} Seconds")
