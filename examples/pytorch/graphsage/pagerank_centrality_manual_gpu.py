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

def pagerank_manual_power_iteration(g, damping_factor=0.85, max_iter=100, tol=1e-6):
    """
    Calculates PageRank using a manual power iteration loop on the GPU with CuPy.
    This implementation is more explicit about the GPU operations.
    """
    if g.device.type != 'cuda':
        g = g.to('cuda')

    num_nodes = g.num_nodes()
    
    # DGL's CSR represents the graph transpose (A^T), where edges are v->u for adj[u,v]
    indptr, indices, _ = g.adj_tensors('csr')
    indptr_cp = cp.asarray(indptr.long())
    indices_cp = cp.asarray(indices.long())
    data_cp = cp.ones(len(indices_cp), dtype=cp.float32)
    A_T = csr_matrix((data_cp, indices_cp, indptr_cp), shape=(num_nodes, num_nodes))

    # Calculate out-degrees for the original graph A
    out_degrees = g.out_degrees().float().to(g.device)
    out_degrees_cp = cp.asarray(out_degrees)

    # Power Iteration
    p = cp.ones(num_nodes, dtype=cp.float32) / num_nodes
    
    dangling_nodes_mask = (out_degrees_cp == 0)
    
    teleport_val = (1 - damping_factor) / num_nodes

    for i in range(max_iter):
        p_old = p.copy()

        # 1. Calculate rank contribution from non-dangling nodes
        # Create a vector of p[j]/deg[j]
        p_div_deg = p / out_degrees_cp
        p_div_deg[dangling_nodes_mask] = 0.0  # Set contribution from dangling nodes to 0
        
        # Multiply by A^T to get sum over incoming neighbors
        p_new = A_T.dot(p_div_deg)

        # 2. Handle dangling nodes' rank redistribution
        # Total rank from all dangling nodes
        dangling_rank_sum = cp.sum(p[dangling_nodes_mask])
        # Distribute it evenly
        p_new += dangling_rank_sum / num_nodes

        # 3. Apply damping factor and teleportation
        p_new = teleport_val + damping_factor * p_new
        
        # 4. Check for convergence
        err = cp.linalg.norm(p_new - p_old)
        if err < num_nodes * tol:
            print(f"Converged after {i+1} iterations.")
            p = p_new
            break
        p = p_new
    else:
        print(f"Did not converge within {max_iter} iterations.")

    return th.as_tensor(p, device='cuda')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-arxiv",
        help="Dataset name ('cora', 'flickr', 'reddit', 'ogbn-products', etc.)"
    )
    parser.add_argument(
        "--damping_factor",
        type=float,
        default=0.85,
        help="Damping factor for PageRank."
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum iterations for PageRank."
    )
    args = parser.parse_args()
    print(f"Calculating PageRank Centrality for {args.dataset} using manual GPU implementation.")

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
    
    if G.device.type != 'cuda':
        G = G.to('cuda')
    
    total_start = time.time()

    # Calculate PageRank using the manual GPU function
    pagerank_th = pagerank_manual_power_iteration(G, damping_factor=args.damping_factor, max_iter=args.max_iter)
    pagerank_cp = cp.asarray(pagerank_th)
    
    # --- Sort column indices based on centrality ---
    row_ptr_cp = cp.asarray(G.adj_tensors('csr')[0].long())
    col_idx_cp = cp.asarray(G.adj_tensors('csr')[1].long())
    
    num_nodes = G.num_nodes()
    sorted_col_idx_cp = cp.empty_like(col_idx_cp)

    for u in range(num_nodes):
        start, end = row_ptr_cp[u], row_ptr_cp[u+1]
        neighbors = col_idx_cp[start:end]

        if neighbors.size > 0:
            order = cp.argsort(-pagerank_cp[neighbors])
            sorted_col_idx_cp[start:end] = neighbors[order]

    # --- ensure output folder exists ---
    out_dir = "pagerank-centrality-manual-gpu"
    os.makedirs(out_dir, exist_ok=True)

    # --- save results ---
    filename_centrality = os.path.join(out_dir, f"{args.dataset}_pagerank-manual-centrality.txt")
    filename_sorted_col_idx = os.path.join(out_dir, f"{args.dataset}_pagerank-manual_sorted-col-index.txt")
    
    # Transfer data back to CPU for saving
    np.savetxt(filename_centrality, pagerank_cp.get(), fmt="%.10f")
    np.savetxt(filename_sorted_col_idx, sorted_col_idx_cp.get(), fmt="%d")
    
    print(f"✅ PageRank centrality saved to {filename_centrality}")
    print(f"✅ Sorted column-index saved to {filename_sorted_col_idx}")
    
    total_end = time.time()
    print(f"Total time: {total_end - total_start:.4f} Seconds")
