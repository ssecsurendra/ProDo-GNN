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

def eigenvector_centrality_power_iteration(g, max_iter=100, tol=1e-6):
    """
    Calculates eigenvector centrality using the power iteration method on a GPU.
    """
    if g.device.type != 'cuda':
        g = g.to('cuda')
    indptr, indices, _ = g.adj_tensors('csr')
    indptr_cp = cp.asarray(indptr.long())
    indices_cp = cp.asarray(indices.long())
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
            print(f"Eigenvector centrality converged after {i+1} iterations.")
            break
        x_old = cp.copy(x)
    else:
        print(f"Eigenvector centrality did not converge within {max_iter} iterations.")
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv", help="Dataset name")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum iterations for power iteration")
    args = parser.parse_args()
    print(f"Calculating Hybrid1 (0.4*Degree + 0.6*Eigen) Centrality for {args.dataset}.")

    # --- GPU Check ---
    if not th.cuda.is_available():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! GPU not available, running on CPU.   !!!")
        print("!!! This will be VERY SLOW for large     !!!")
        print("!!! datasets like Reddit.                !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        device = 'cpu'
    else:
        device = 'cuda'
        print("--- GPU available. Running on GPU. ---")


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
        elif args.dataset == "igb-medium":
            load_path = './dataset/igb_medium.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "igb-tiny":
            load_path = './dataset/igb_tiny.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "amazon_products":
            load_path = './dataset/amazon_products.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "cit-net":
            load_path = '/data/Dataset/gnn_dataset/citations_network_graph.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "wiki":
            load_path = './dataset/wikidata5M/wikidata5m_dgl_graph.bin'
            data, _ = dgl.load_graphs(load_path)
        else:
            raise ValueError("Unknown dataset: {}".format(args.dataset))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    G = data[0]
    G = dgl.to_bidirected(G, copy_ndata=True)
    print(f"Moving graph to {device}...")
    G = G.to(device)
    print(f"Graph is on device: {G.device}")
    
    total_start = time.time()

    # 1. Calculate Degree Centrality
    print("Calculating Degree Centrality...")
    degree_centrality = G.in_degrees().float()
    degree_centrality_cp = cp.asarray(degree_centrality)
    print("Done.")

    # 2. Calculate Eigenvector Centrality
    print("Calculating Eigenvector Centrality...")
    eigen_centrality_cp = eigenvector_centrality_power_iteration(G, max_iter=args.max_iter)
    print("Done.")

    # 3. Normalize both centralities to be in [0, 1]
    max_deg = cp.max(degree_centrality_cp)
    if max_deg > 0:
        degree_norm = degree_centrality_cp / max_deg
    else:
        degree_norm = degree_centrality_cp

    max_eig = cp.max(eigen_centrality_cp)
    if max_eig > 0:
        eigen_norm = eigen_centrality_cp / max_eig
    else:
        eigen_norm = eigen_centrality_cp

    # 4. Compute weighted hybrid centrality
    print("Combining centralities...")
    hybrid1_centrality_cp = 0.4 * degree_norm + 0.6 * eigen_norm
    print("Done.")

    # 5. Sort column indices
    row_ptr_cp = cp.asarray(G.adj_tensors('csr')[0].long())
    col_idx_cp = cp.asarray(G.adj_tensors('csr')[1].long())
    
    num_nodes = G.num_nodes()
    sorted_col_idx_cp = cp.empty_like(col_idx_cp)

    for u in range(num_nodes):
        start, end = row_ptr_cp[u], row_ptr_cp[u+1]
        neighbors = col_idx_cp[start:end]

        if neighbors.size > 0:
            order = cp.argsort(-hybrid1_centrality_cp[neighbors])
            sorted_col_idx_cp[start:end] = neighbors[order]

    # 6. Save results
    out_dir = "hybrid1-centrality-gpu"
    os.makedirs(out_dir, exist_ok=True)

    filename_centrality = os.path.join(out_dir, f"{args.dataset}_hybrid1-centrality.txt")
    filename_sorted_col_idx = os.path.join(out_dir, f"{args.dataset}_hybrid1_sorted-col-index.txt")
    
    np.savetxt(filename_centrality, hybrid1_centrality_cp.get(), fmt="%.10f")
    np.savetxt(filename_sorted_col_idx, sorted_col_idx_cp.get(), fmt="%d")
    
    print(f"✅ Hybrid1 centrality saved to {filename_centrality}")
    print(f"✅ Sorted column-index saved to {filename_sorted_col_idx}")
    
    total_end = time.time()
    print(f"Total time: {total_end - total_start:.4f} Seconds")
