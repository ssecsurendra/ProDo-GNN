
import dgl
import torch
import time
import argparse
import numpy as np
import dgl.function as fn
from dgl.data import RedditDataset, YelpDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset

def get_degree_sorted_nodes(g):
    """
    Sorts nodes by their out-degree in descending order.
    This is used as a heuristic for the greedy dominating set algorithm.
    """
    print("Sorting nodes by degree...")
    degrees = g.out_degrees().to(torch.float32)
    sorted_nodes = torch.argsort(degrees, descending=True)
    print("Node sorting complete.")
    return sorted_nodes

def optimized_k_dominating_set(g, k):
    """
    Computes a distance-k dominating set for the entire graph using a parallel
    message-passing algorithm. The greedy choice is based on node degree.
    """
    device = g.device
    num_nodes = g.num_nodes()
    
    sorted_nodes = get_degree_sorted_nodes(g)
    
    print("Creating position map...")
    large_float = float(num_nodes)
    pos_map = torch.full((num_nodes,), large_float, dtype=torch.float32, device=device)
    pos_map[sorted_nodes] = torch.arange(num_nodes, device=device, dtype=torch.float32)

    g.ndata['min_rank'] = pos_map
    print(f"Starting message passing for k={k} hops...")

    # Propagate the best rank (min_rank) for k hops
    for i in range(k):
        # Each node sends its current best rank to its neighbors.
        g.update_all(fn.copy_u('min_rank', 'm'), fn.min('m', 'neighbor_min_rank'))
        # Each node updates its best rank by choosing the minimum between its
        # current rank and the best rank it received from its neighbors.
        g.ndata['min_rank'] = torch.min(g.ndata['min_rank'], g.ndata['neighbor_min_rank'])
        print(f"Completed hop {i+1}/{k}...")

    final_min_pos = g.ndata['min_rank'].long()
    
    # Clean up intermediate node data
    del g.ndata['min_rank']
    del g.ndata['neighbor_min_rank']

    # The final dominating nodes are the ones with the best rank in their k-hop neighborhood
    dominating_nodes = sorted_nodes[final_min_pos]
    
    # The final dominating set is the unique collection of these "winner" nodes.
    dominating_set = torch.unique(dominating_nodes)
    
    return dominating_set

def load_graph(dataset_name, device):
    """
    Loads the specified dataset and moves it to the given device.
    """
    print(f"Loading {dataset_name} dataset...")
    if dataset_name == 'reddit':
        dataset = RedditDataset()
        g = dataset[0]
    elif dataset_name == 'ogbn-products':
        dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
        g = dataset[0]
    elif dataset_name == 'ogbn-arxiv':
        dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv"))
        g = dataset[0]
    elif dataset_name == 'yelp':
        dataset = YelpDataset()
        g = dataset[0]
    elif dataset_name == 'igb-small':
        load_path = './dataset/igb_small.dgl'
        graphs, _ = dgl.load_graphs(load_path)
        g = graphs[0]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # The algorithm works on an undirected graph.
    g = dgl.to_bidirected(g, copy_ndata=True)
    g = g.to(device)
    print("Dataset loaded successfully.")
    return g

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Distance-k Dominating Set for an Entire Graph")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=['reddit', 'ogbn-products', 'ogbn-arxiv', 'yelp', 'igb-small'],
                        help="Name of the dataset to process.")
    parser.add_argument("--k", type=int, required=True, help="The distance k for the dominating set.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load graph
    g = load_graph(args.dataset, device)
    print(f"Graph for {args.dataset}: {g.num_nodes()} nodes, {g.num_edges()} edges.")

    # Compute dominating set and measure time
    print(f"Starting k-dominating set computation for k={args.k}...")
    start_time = time.time()
    dom_set = optimized_k_dominating_set(g, args.k)
    end_time = time.time()
    
    computation_time = end_time - start_time
    
    # Save the result
    output_filename = f"{args.dataset}_k{args.k}_dominating_set.npy"
    np.save(output_filename, dom_set.cpu().numpy())

    # Print results
    print("\n--- Results ---")
    print(f"k-Dominating set computation time (k={args.k}): {computation_time:.4f} seconds")
    print(f"Number of nodes in the dominating set: {len(dom_set)}")
    print(f"Dominating set saved to: {output_filename}")
