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

def eigenvector_centrality_power_iteration(g, initial_vector, max_iter=100, tol=1e-6):
    """
    Calculates eigenvector centrality using the power iteration method on a GPU,
    starting from an initial vector.

    Parameters:
    g (DGLGraph): The input graph.
    initial_vector (cupy.ndarray): The initial vector for power iteration.
    max_iter (int): Maximum number of iterations for power iteration.
    tol (float): Tolerance to check for convergence.

    Returns:
    cupy.ndarray: The eigenvector centrality scores for each node.
    """
    # Ensure the graph is on the GPU
    if g.device.type != 'cuda':
        g = g.to('cuda')

    # Get the CSR representation of the graph's adjacency matrix
    indptr, indices, _ = g.adj_tensors('csr')
    
    # Transfer to CuPy arrays
    indptr_cp = cp.asarray(indptr.long())
    indices_cp = cp.asarray(indices.long())
    
    # Create a CuPy sparse matrix
    num_nodes = g.num_nodes()
    # Assuming unweighted graph, so data is all ones
    data_cp = cp.ones(len(indices_cp), dtype=cp.float32)
    adj_matrix = csr_matrix((data_cp, indices_cp, indptr_cp), shape=(num_nodes, num_nodes))

    # Power Iteration
    x = cp.copy(initial_vector) # Use the provided initial vector
    x_old = cp.copy(x)

    for i in range(max_iter):
        x = adj_matrix.dot(x)
        norm = cp.linalg.norm(x)
        if norm == 0:
            return x # Return zero vector if norm is zero
        x /= norm

        # Check for convergence
        if cp.linalg.norm(x - x_old) < tol:
            print(f"Converged after {i+1} iterations.")
            break
        x_old = cp.copy(x)
    else: # This else belongs to the for loop, and executes if the loop finishes without a 'break'
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
    print(f"Calculating Hybrid Centrality for {args.dataset} using GPU.")

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
    
    # Ensure the graph is undirected for eigenvector centrality
    G = dgl.to_bidirected(G, copy_ndata=True)

    if G.device.type != 'cuda':
        G = G.to('cuda')
    
    total_start = time.time()

    # 1. Calculate degree centrality to use as the initial vector
    # Using in_degrees which is equivalent to out_degrees for an undirected graph
    degrees = G.in_degrees().float()
    # Add 1 to avoid zero degrees, which would result in a zero initial vector for isolated nodes
    degrees += 1.0 
    initial_vector_th = degrees
    initial_vector_cp = cp.asarray(initial_vector_th)

    # 2. Calculate Eigenvector Centrality using GPU with degree centrality as initial vector
    hybrid_centrality_cp = eigenvector_centrality_power_iteration(G, initial_vector=initial_vector_cp, max_iter=args.max_iter)
    
    # --- Sort column indices based on centrality ---
    row_ptr_cp = cp.asarray(G.adj_tensors('csr')[0].long())
    col_idx_cp = cp.asarray(G.adj_tensors('csr')[1].long())
    
    num_nodes = G.num_nodes()
    sorted_col_idx_cp = cp.empty_like(col_idx_cp)

    for u in range(num_nodes):
        start, end = row_ptr_cp[u], row_ptr_cp[u+1]
        neighbors = col_idx_cp[start:end]

        if neighbors.size > 0:
            order = cp.argsort(-hybrid_centrality_cp[neighbors])
            sorted_col_idx_cp[start:end] = neighbors[order]

    # --- ensure output folder exists ---
    out_dir = "hybrid-centrality-gpu"
    os.makedirs(out_dir, exist_ok=True)

    # --- save results ---
    filename_centrality = os.path.join(out_dir, f"{args.dataset}_hybrid-centrality.txt")
    filename_sorted_col_idx = os.path.join(out_dir, f"{args.dataset}_hybrid_sorted-col-index.txt")
    
    # Transfer data back to CPU for saving
    np.savetxt(filename_centrality, hybrid_centrality_cp.get(), fmt="%.10f")
    np.savetxt(filename_sorted_col_idx, sorted_col_idx_cp.get(), fmt="%d")
    
    print(f"✅ Hybrid centrality saved to {filename_centrality}")
    print(f"✅ Sorted column-index saved to {filename_sorted_col_idx}")
    
    total_end = time.time()
    print(f"Total time: {total_end - total_start:.4f} Seconds")
