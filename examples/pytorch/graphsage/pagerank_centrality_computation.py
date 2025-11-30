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
    print(f"Calculating PageRank Centrality for {args.dataset} using GPU.")

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
    
    # PageRank is typically used on directed graphs, but making it bidirected is common for undirected graphs.
    G = dgl.to_bidirected(G, copy_ndata=True)
    
    if G.device.type != 'cuda':
        G = G.to('cuda')
    
    total_start = time.time()

    # Calculate PageRank using DGL's built-in function
    pagerank_th = dgl.pagerank(G, damping_factor=args.damping_factor, max_iter=args.max_iter)
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
    out_dir = "pagerank-centrality-gpu"
    os.makedirs(out_dir, exist_ok=True)

    # --- save results ---
    filename_centrality = os.path.join(out_dir, f"{args.dataset}_pagerank-centrality.txt")
    filename_sorted_col_idx = os.path.join(out_dir, f"{args.dataset}_pagerank_sorted-col-index.txt")
    
    # Transfer data back to CPU for saving
    np.savetxt(filename_centrality, pagerank_cp.get(), fmt="%.10f")
    np.savetxt(filename_sorted_col_idx, sorted_col_idx_cp.get(), fmt="%d")
    
    print(f"✅ PageRank centrality saved to {filename_centrality}")
    print(f"✅ Sorted column-index saved to {filename_sorted_col_idx}")
    
    total_end = time.time()
    print(f"Total time: {total_end - total_start:.4f} Seconds")
