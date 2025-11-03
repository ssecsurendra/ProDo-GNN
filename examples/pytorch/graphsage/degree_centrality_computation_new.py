import numpy as np
import pandas as pd
import sys
import dgl
import time
import os
import torch as th
#import pymetis
#import gc
from scipy.io import mmread
import os
from tqdm import tqdm
import cupy as cp
import psutil
import argparse
os.environ["DGLBACKEND"] = "pytorch"
import torch.nn.functional as F
import torch
import math
import copy
import random
import dgl.data
# from . import backend as F, utils
from dgl import AddSelfLoop
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, WisconsinDataset, FlickrDataset, RedditDataset, YelpDataset

# CUDA kernel for degree centrality
degree_kernel = cp.RawKernel(r'''
extern "C" __global__
void degree_centrality(const int* row_ptr, float* centrality, int num_nodes) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //printf("Launching kernel on GPU...\n");
    if (tid < num_nodes) {
        int degree = row_ptr[tid + 1] - row_ptr[tid];
        centrality[tid] = (float)degree / (num_nodes - 1);
    }
}
''', 'degree_centrality')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-arxiv",
    )
    # parser.add_argument(
    #     "--num_clusters",
    #     type=int,
    #     default=20,
    #     help="Number of clusters",
    # )
    # parser.add_argument(
    #     "--dt",
    #     type=str,
    #     default="float",
    #     help="data type (float, bfloat16)",
    # )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")

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
        elif args.dataset == "wiki":
            load_path = './dataset/wikidata5M/wikidata5m_dgl_graph.bin'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "igb-small":
            load_path = './dataset/igb_small.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "amazon_products":
            load_path = './dataset/amazon_products.dgl'
            data, _ = dgl.load_graphs(load_path)       
        else:
            raise ValueError("Unknown dataset: {}".format(args.dataset))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    G = data[0]
        # ---------------------- DGL PREPROCESS -----------------------------------#
    total_start = time.time()
    # ensure undirected
    #G = dgl.to_bidirected(G, copy_ndata=True)
    row_ptr = np.array(G.adj_tensors('csr')[0])
    col_idx = np.array(G.adj_tensors('csr')[1])
    row_ptr = cp.asarray(row_ptr, dtype=cp.int32)
    col_idx = cp.asarray(col_idx, dtype=cp.int32)
     # basic checks
    assert isinstance(row_ptr, cp.ndarray) and row_ptr.dtype == cp.int32
    assert isinstance(col_idx, cp.ndarray) and col_idx.dtype == cp.int32

    num_nodes = int(row_ptr.size - 1)
    num_edges = int(col_idx.size)
    #num_nodes = row_ptr.size - 1

    # Allocate output
    centrality = cp.zeros(num_nodes, dtype=cp.float32)

    # Kernel launch configuration
    threads_per_block = 256
    blocks = (num_nodes + threads_per_block - 1) // threads_per_block

    # Launch kernel
    degree_kernel((blocks,), (threads_per_block,), 
                  (row_ptr, centrality, num_nodes))

    cp.cuda.Device(0).synchronize()

    # --- 2) build row_ids (segmented indices) on GPU ---
    # degrees: length num_nodes
    degrees = (row_ptr[1:] - row_ptr[:-1]).astype(cp.int32)

    # If graph has zero edges, return early
    # if num_edges == 0:
    #     return cp.array([], dtype=cp.int32), centrality

    # row_ids length = num_edges, row_ids[i] = u where adjacency entry i belongs to node u
    # cp.repeat works on GPU
    node_ids = cp.arange(num_nodes, dtype=cp.int32)
    row_ids = cp.repeat(node_ids, degrees)   # shape == num_edges
    # now row_ids[i] tells which source node the i-th adjacency belongs to

    # --- 3) compute secondary key: -centrality[neighbor]  (so lexsort gives descending)
    neighbor_centrality_neg = -centrality[col_idx]  # shape == num_edges, GPU array

    # --- 4) lexsort: primary key = row_ids, secondary key = neighbor_centrality_neg
    # numpy/cupy.lexsort takes keys sequence where last key is primary.
    # We want primary=row_ids, secondary=neighbor_centrality_neg.
    # So pass (neighbor_centrality_neg, row_ids)
    # Result: indices of length num_edges that sort by row_ids first then -centrality within each row.
    perm = cp.lexsort((neighbor_centrality_neg, row_ids))  # GPU operation

    # --- 5) apply permutation to col_idx to get sorted flattened adjacency list ---
    sorted_col_idx = col_idx[perm]
    cp.asnumpy(sorted_col_idx).astype('int32')
    # # Allocate output
    # sorted_col_idx = cp.empty_like(col_idx)
    #
    # # Sort neighbors of each node
    # for u in range(num_nodes):
    #     start, end = row_ptr[u], row_ptr[u+1]
    #     neighbors = col_idx[start:end]
    #
    #     if neighbors.size > 0:
    #         # Sort neighbors by descending degree centrality
    #         order = cp.argsort(-centrality[neighbors])
    #         sorted_col_idx[start:end] = neighbors[order]
    #
    # # --- ensure output folder exists ---
    # out_dir = "degree-centrality"
    # os.makedirs(out_dir, exist_ok=True)

    # --- save with dataset name inside folder ---
    # filename = os.path.join(out_dir, f"{args.dataset}_degree-centrality.txt")
    filename1 = (f"{args.dataset}_sorted-col-index-new.txt")
    filename2 = (f"{args.dataset}_degree-centrality-new.txt")
    cp.savetxt(filename2, centrality.get(), fmt="%.10f")
    cp.savetxt(filename1, sorted_col_idx.get(), fmt="%d")
    print(f"✅ Degree centrality saved to {filename2}")
    print(f"✅ sorted column-index saved to {filename1}")
    #print("Degree centrality:\n", centrality)
    total_end = time.time()
    print("Total time", total_end-total_start, "Seconds")
    
         
