import numpy as np
import pandas as pd
import sys
import dgl
import time
import os
import torch as th
import cupy as cp
import argparse
os.environ["DGLBACKEND"] = "pytorch"
import torch.nn.functional as F
import torch
import math
import copy
import random
import dgl.data
import networkx as nx
from dgl import AddSelfLoop
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, WisconsinDataset, FlickrDataset, RedditDataset, YelpDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-arxiv",
    )
    args = parser.parse_args()
    print(f"Calculating Eigenvector Centrality for {args.dataset}.")

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
        elif args.dataset == "amazon_products":
            load_path = '/data/Dataset/gnn_dataset/amazon_products.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "cit-net":
            load_path = '/data/Dataset/gnn_dataset/citations_network_graph.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "igb-tiny":
            load_path = './dataset/igb_tiny.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "igb-medium":
            load_path = './dataset/igb_medium.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "igb-small":
            load_path = './dataset/igb_small.dgl'
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
    total_start = time.time()

    # Calculate Eigenvector Centrality
    # Ensure graph is on CPU for eigenvector_centrality if it's not already
    if G.device.type == 'cuda':
        G = G.to('cpu')
    
    # Manually construct a NetworkX simple graph
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(G.num_nodes()))
    u, v = G.edges()
    for i in range(len(u)):
        nx_graph.add_edge(u[i].item(), v[i].item())
    
    # Calculate eigenvector centrality using NetworkX
    eigen_centrality_dict = nx.eigenvector_centrality(nx_graph)
    
    # Convert the dictionary to a NumPy array, ensuring order by node ID
    eigen_centrality_np = np.array([eigen_centrality_dict[i] for i in range(G.num_nodes())])

    num_nodes = G.num_nodes()
    
    # Get row_ptr and col_idx for sorting neighbors
    # Ensure these are on CPU if the graph was moved there
    row_ptr = G.adj_tensors('csr')[0].numpy()
    col_idx = G.adj_tensors('csr')[1].numpy()

    # Allocate output for sorted column indices
    sorted_col_idx = np.empty_like(col_idx)

    # Sort neighbors of each node by descending eigenvector centrality
    for u in range(num_nodes):
        start, end = row_ptr[u], row_ptr[u+1]
        neighbors = col_idx[start:end]

        if neighbors.size > 0:
            # Sort neighbors by descending eigenvector centrality
            order = np.argsort(-eigen_centrality_np[neighbors])
            sorted_col_idx[start:end] = neighbors[order]

    # --- ensure output folder exists ---
    out_dir = "eigenvector-centrality" # New directory for eigenvector centrality
    os.makedirs(out_dir, exist_ok=True)

    # --- save with dataset name inside folder ---
    filename_centrality = os.path.join(out_dir, f"{args.dataset}_eigen-centrality.txt")
    filename_sorted_col_idx = os.path.join(out_dir, f"{args.dataset}_sorted-col-index.txt")
    
    np.savetxt(filename_centrality, eigen_centrality_np, fmt="%.10f")
    np.savetxt(filename_sorted_col_idx, sorted_col_idx, fmt="%d")
    
    print(f"✅ Eigenvector centrality saved to {filename_centrality}")
    print(f"✅ Sorted column-index saved to {filename_sorted_col_idx}")
    
    total_end = time.time()
    print("Total time", total_end-total_start, "Seconds")
