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
import matplotlib.pyplot as plt
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
        elif args.dataset == "igb-small":
            load_path = './dataset/igb_small.dgl'
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

    g = data[0]
    dataset_name = str(args.dataset)
    labels = g.ndata['label']  # node labels
    #print("length of labels: ",len(labels))
    train_mask = g.ndata['train_mask']  # boolean mask for training nodes

    # Select only training nodes
    train_labels = labels[train_mask]

    # Count number of nodes per class
    unique_labels, counts = torch.unique(train_labels, return_counts=True)
    # Convert to CPU and Python types for plotting
    classes = unique_labels.cpu().tolist()
    node_counts = counts.cpu().tolist()
    print(f"Dataset Name: {dataset_name}")
    print("Class distribution:")
    print("Sum of all elements in label: ",sum(node_counts))
    # Compute total and percentage
    total_train_nodes = sum(node_counts)
    percentages = [(cnt / total_train_nodes) * 100 for cnt in node_counts]

    for cls, cnt, pct in zip(classes, node_counts, percentages):
         print(f"  Class {cls}: {cnt} nodes ({pct:.2f}%)")
    print(f"\nTotal Nodes: {len(labels)}")
    # --- Plot class distribution ---
    plt.figure(figsize=(8, 5))
    plt.bar(classes, node_counts, color='steelblue', edgecolor='black')
    plt.xlabel("Class Label", fontsize=12)
    plt.ylabel("Number of Nodes", fontsize=12)
    plt.title(f"Class Distribution in {dataset_name.capitalize()} Dataset", fontsize=14)
    plt.xticks(classes)
    plt.tight_layout()

    # --- Save as EPS file ---
    plot_name= str(args.dataset)+"_class_distribution.eps"
    plt.savefig(plot_name, format='eps', dpi=300)
    plt.show()

    # --- Plot 2: Percentages ---
    plt.figure(figsize=(8, 5))
    plt.bar(classes, percentages, color='orange', edgecolor='black')
    plt.xlabel("Class Label", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.title(f"Training Class Distribution (Percentage) - {dataset_name.capitalize()}", fontsize=14)
    plt.xticks(classes)
    plt.tight_layout()
    plot_name1= str(args.dataset)+"_class_distribution_percentage.eps"
    plt.savefig(plot_name1, format='eps', dpi=300)
    plt.show()

    print("\nPlot are saved")
    # ---------------------- DGL PREPROCESS -----------------------------------#
    # total_start = time.time()
    # # ensure undirected
    # #G = dgl.to_bidirected(G, copy_ndata=True)
    # row_ptr = np.array(G.adj_tensors('csr')[0])
    # col_idx = np.array(G.adj_tensors('csr')[1])
    # row_ptr = cp.asarray(row_ptr, dtype=cp.int32)
    # col_idx = cp.asarray(col_idx, dtype=cp.int32)
    # num_nodes = row_ptr.size - 1
    #
    # # Allocate output
    # centrality = cp.zeros(num_nodes, dtype=cp.float32)
    #
    # # Kernel launch configuration
    # threads_per_block = 256
    # blocks = (num_nodes + threads_per_block - 1) // threads_per_block
    #
    # # Launch kernel
    # degree_kernel((blocks,), (threads_per_block,), 
    #               (row_ptr, centrality, num_nodes))
    #
    # cp.cuda.Device(0).synchronize()
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
    #
    # # --- save with dataset name inside folder ---
    # # filename = os.path.join(out_dir, f"{args.dataset}_degree-centrality.txt")
    # filename1 = (f"{args.dataset}_sorted-col-index.txt")
    # filename2 = (f"{args.dataset}_degree-centrality.txt")
    # cp.savetxt(filename2, centrality.get(), fmt="%.10f")
    # cp.savetxt(filename1, sorted_col_idx.get(), fmt="%d")
    # print(f"✅ Degree centrality saved to {filename2}")
    # print(f"✅ sorted column-index saved to {filename1}")
    # #print("Degree centrality:\n", centrality)
    # total_end = time.time()
    # print("Total time", total_end-total_start, "Seconds")
    
         
