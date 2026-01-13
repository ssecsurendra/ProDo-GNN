
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
    degrees = g.out_degrees().to(torch.float32)
    sorted_nodes = torch.argsort(degrees, descending=True)
    return sorted_nodes

def optimized_dominating_set(g):
    """
    Computes a dominating set for the entire graph using a parallel
    message-passing algorithm. The greedy choice is based on node degree.
    """
    device = g.device
    num_nodes = g.num_nodes()
    
    # Get nodes sorted by degree, which defines their importance
    sorted_nodes = get_degree_sorted_nodes(g)
    
    # Create a position map where lower value means higher importance
    large_float = float(num_nodes)
    pos_map = torch.full((num_nodes,), large_float, dtype=torch.float32, device=device)
    pos_map[sorted_nodes] = torch.arange(num_nodes, device=device, dtype=torch.float32)

    # Each node starts with its own importance value
    g.ndata['min_pos'] = pos_map

    # DGL message passing: each node receives the importance value from all its
    # neighbors and finds the minimum value among them.
    g.update_all(fn.copy_u('min_pos', 'm'), fn.min('m', 'min_neighbor_pos'))
    
    # The final "dominator" for a node is the one with the best rank (lowest value)
    # in its 1-hop neighborhood (including itself).
    final_min_pos = torch.min(g.ndata['min_pos'], g.ndata['min_neighbor_pos']).long()

    # Look up the actual node IDs from their ranks
    dominating_nodes = sorted_nodes[final_min_pos]
    
    # The final dominating set is the unique set of these "winner" nodes.
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
        # Path from the initial context
        load_path = './dataset/igb_small.dgl'
        graphs, _ = dgl.load_graphs(load_path)
        g = graphs[0]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    g = g.to(device)
    print("Dataset loaded successfully.")
    return g

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Dominating Set for an Entire Graph")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=['reddit', 'ogbn-products', 'ogbn-arxiv', 'yelp', 'igb-small'],
                        help="Name of the dataset to process.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load graph
    g = load_graph(args.dataset, device)
    print(f"Graph for {args.dataset}: {g.num_nodes()} nodes, {g.num_edges()} edges.")

    # Compute dominating set and measure time
    print("Starting dominating set computation...")
    start_time = time.time()
    dom_set = optimized_dominating_set(g)
    end_time = time.time()
    
    computation_time = end_time - start_time
    
    # Save the result
    output_filename = f"{args.dataset}_dominating_set.npy"
    np.save(output_filename, dom_set.cpu().numpy())

    # Print results
    print("\n--- Results ---")
    print(f"Dominating set computation time: {computation_time:.4f} seconds")
    print(f"Number of nodes in the dominating set: {len(dom_set)}")
    print(f"Dominating set saved to: {output_filename}")
