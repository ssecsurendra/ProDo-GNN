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

# --- Kernel 1: CUDA kernel for degree centrality ---
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

# --- Kernel 2: Sort neighbors by degree centrality ---
neighbor_sort_kernel = cp.RawKernel(r'''
extern "C" __global__
void sort_neighbors(const int* row_ptr,
                    const int* col_idx,
                    const float* centrality,
                    int* sorted_col_idx,
                    int num_nodes) {
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    if (u >= num_nodes) return;

    int start = row_ptr[u];
    int end   = row_ptr[u+1];
    int degree = end - start;

    // bubble sort (ok for small degree)
    for (int i = 0; i < degree; i++) {
        for (int j = i + 1; j < degree; j++) {
            int ni = col_idx[start + i];
            int nj = col_idx[start + j];
            if (centrality[ni] < centrality[nj]) {
                int tmp = col_idx[start + i];
                col_idx[start + i] = col_idx[start + j];
                col_idx[start + j] = tmp;
            }
        }
    }

    for (int i = 0; i < degree; i++) {
        sorted_col_idx[start + i] = col_idx[start + i];
    }
}
''', 'sort_neighbors')


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
    num_nodes = row_ptr.size - 1

    # Allocate output
    centrality = cp.zeros(num_nodes, dtype=cp.float32)

    # Kernel launch configuration
    threads_per_block = 256
    blocks = (num_nodes + threads_per_block - 1) // threads_per_block

    # Launch kernel 1: degree centrality
    degree_kernel((blocks,), (threads_per_block,), 
                  (row_ptr, centrality, num_nodes))

    cp.cuda.Device(0).synchronize()
    # Allocate output
    sorted_col_idx = cp.empty_like(col_idx)
    # --- Launch kernel 2: sort neighbors ---
    neighbor_sort_kernel((blocks,), (threads_per_block,), (row_ptr, col_idx, centrality, sorted_col_idx, num_nodes))

    cp.cuda.Stream.null.synchronize()

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
    filename1 = (f"{args.dataset}_sorted-col-index.txt")
    filename2 = (f"{args.dataset}_degree-centrality.txt")
    cp.savetxt(filename2, centrality.get(), fmt="%.10f")
    cp.savetxt(filename1, sorted_col_idx.get(), fmt="%d")
    print(f"✅ Degree centrality saved to {filename2}")
    print(f"✅ sorted column-index saved to {filename1}")
    #print("Degree centrality:\n", centrality)
    total_end = time.time()
    print("Total time", total_end-total_start, "Seconds")
    
         
