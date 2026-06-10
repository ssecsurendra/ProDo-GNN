import dgl
import argparse
import time
import matplotlib.pyplot as plt
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
# from dgl.sampling.metis_sampling import *
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
#import dgl.data.CoraGraphDataset
from dgl.data import CoraGraphDataset,RedditDataset,FlickrDataset,YelpDataset
import numpy as np
import cupy as cp


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h) 
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, graph, dataloader, num_classes, device, feat_cache, node_map, cached_nodes_mask):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            current_input_nodes = blocks[0].srcdata[dgl.NID]
            if feat_cache is not None:
                is_cached = cached_nodes_mask[current_input_nodes]
                cached_part = current_input_nodes[is_cached]
                non_cached_part = current_input_nodes[~is_cached]

                x = torch.empty(len(current_input_nodes), graph.ndata['feat'].shape[1], dtype=graph.ndata['feat'].dtype, device=device)

                if len(cached_part) > 0:
                    cached_indices_in_cache = node_map[cached_part]
                    x[is_cached] = feat_cache[cached_indices_in_cache]

                if len(non_cached_part) > 0:
                    x[~is_cached] = graph.ndata['feat'][non_cached_part.to('cpu')].to(device)
            else:
                x = graph.ndata['feat'][current_input_nodes].to(device)

            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(
            pred, label, task="multiclass", num_classes=num_classes
        )


def train(args, device, g, dataset, model, num_classes, centrality_vals, feat_cache, node_map, cached_nodes_mask):
    train_mask=g.ndata['train_mask']
    val_mask=g.ndata['val_mask']
    train_idx = torch.nonzero(train_mask).squeeze().to(device)

    train_degrees = centrality_vals[train_idx]
    sorted_idx = cp.argsort(-train_degrees)
    k = int(1.0 * len(train_idx))
    top_train_idx = train_idx[sorted_idx[:k]]
    print("sorted training node: ",top_train_idx)

    N = g.num_nodes()
    is_train = torch.zeros(N, dtype=torch.bool, device='cpu')
    is_train[train_idx.to('cpu')] = True

    remaining = is_train.clone()
    dominating_set_list = []
    indptr, indices, edge_ids = g.adj_tensors('csr')

    dominating_start_time = time.time()
    for u in top_train_idx:
        u = u.item()
        if not remaining[u]:
            continue
        dominating_set_list.append(u)
        start, end = indptr[u], indptr[u + 1]
        nbrs = indices[start:end]
        train_nbrs = nbrs[is_train[nbrs]]
        remaining[u] = False
        remaining[train_nbrs] = False
        if not remaining.any():
            break
    dominating_set = torch.tensor(dominating_set_list, device=device)
    dominating_end_time = time.time()
    print("Dominating set computation time: ",dominating_end_time - dominating_start_time)
    print("Dominating set:",dominating_set)
    print("Original training nodes:", len(train_idx))
    print("Number of nodes in dominating set:", len(dominating_set))

    target_train_size = int(args.target_train_percentage * len(train_idx))
    final_train_idx_set = set(dominating_set.cpu().numpy())
    Maintaining_training_start = time.time()
    if len(final_train_idx_set) < target_train_size:
        for node in top_train_idx:
            if len(final_train_idx_set) >= target_train_size:
                break
            final_train_idx_set.add(node.item())
    maintaining_training_end = time.time()
    print("Training node maintaining time: ", maintaining_training_end - Maintaining_training_start)
    final_train_idx = torch.tensor(list(final_train_idx_set))
    print(f"Final training nodes (Dominating Set + Top Centrality): {len(final_train_idx)}")
    val_idx = torch.nonzero(val_mask).squeeze()

    sampler = NeighborSampler(
        [int(fanout) for fanout in args.fanout.split(",")],
        prefetch_labels=["label"],
    )

    use_uva = False
    train_dataloader = DataLoader(
        g, final_train_idx, sampler, device=device, batch_size=args.batch_size,
        shuffle=True, drop_last=False, num_workers=0, use_uva=use_uva
    )
    val_dataloader = DataLoader(
        g, val_idx, sampler, device=device, batch_size=args.batch_size,
        shuffle=True, drop_last=False, num_workers=0, use_uva=use_uva
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    total_training_time = 0.0
    epoch_lines = []

    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        start_time1 = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            current_input_nodes = blocks[0].srcdata[dgl.NID]
            if args.mode == 'mixed' and feat_cache is not None:
                is_cached = cached_nodes_mask[current_input_nodes]
                cached_part = current_input_nodes[is_cached]
                non_cached_part = current_input_nodes[~is_cached]
                x = torch.empty(len(current_input_nodes), g.ndata['feat'].shape[1], dtype=g.ndata['feat'].dtype, device=device)
                if len(cached_part) > 0:
                    cached_indices_in_cache = node_map[cached_part]
                    x[is_cached] = feat_cache[cached_indices_in_cache]
                if len(non_cached_part) > 0:
                    x[~is_cached] = g.ndata['feat'][non_cached_part.to('cpu')].to(device)
            else:
                x = g.ndata['feat'][current_input_nodes].to(device)
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        execution_time = time.time() - start_time1
        total_training_time += execution_time
        
        acc = evaluate(model, g, val_dataloader, num_classes, device, feat_cache, node_map, cached_nodes_mask)
        epoch_line = "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {:.4f}".format(epoch, total_loss / (it + 1), acc.item(), execution_time)
        print(epoch_line)
        epoch_lines.append(epoch_line)

    tt_time = "Total time {:.4f}".format(total_training_time)
    epoch_lines.append(tt_time)
    return epoch_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="mixed", choices=["cpu", "mixed", "puregpu"])
    parser.add_argument("--dataset", type=str, default="reddit")
    parser.add_argument("--fanout", type=str, default="20,20,20")
    parser.add_argument("--target_train_percentage", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--cache_rate", type=float, default=0.5, help="Fraction of nodes to cache based on degree.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.mode = "cpu"
    
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    if args.dataset == "reddit":
        dataset = RedditDataset()
        centrality_file = 'dissimilarity-eigenvector-centrality/reddit_dissimilar_eigen-centrality.npy'
        sortedcol_file = 'dissimilarity-eigenvector-centrality/reddit_dissimilar_eigen_sorted-col-index.npy'
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    G = dataset[0]
    G = G.to("cuda" if args.mode == "puregpu" else "cpu")
    
    indptr, indices, edge_ids = G.adj_tensors('csr')
    
    sorted_col_idx = torch.from_numpy(np.load(sortedcol_file))
    centrality_vals = torch.from_numpy(np.load(centrality_file)).to(device)

    num_nodes = len(indptr) - 1
    g = dgl.graph((torch.searchsorted(indptr[1:], torch.arange(len(indices), device=indptr.device), right=True), sorted_col_idx.to(torch.int64)), num_nodes=num_nodes)
    g.ndata.update(G.ndata)
    del G
    torch.cuda.empty_cache()

    labels = g.ndata["label"]
    num_classes = int(labels.max().item()) + 1
    in_size = g.ndata["feat"].shape[1]
    out_size = num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    feat_cache, node_map, cached_nodes_mask = None, None, None
    if args.mode == 'mixed':
        num_nodes_to_cache = int(g.num_nodes() * args.cache_rate)
        print(f"Caching based on degree: {num_nodes_to_cache} nodes ({args.cache_rate*100}%).")

        if num_nodes_to_cache > 0:
            degrees = g.in_degrees()
            sorted_degrees, sorted_indices = torch.sort(degrees, descending=True)
            high_degree_nodes = sorted_indices[:num_nodes_to_cache]
            
            cached_nodes_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
            cached_nodes_mask[high_degree_nodes] = True
            cached_nodes_mask = cached_nodes_mask.to(device)

            node_map = torch.full((g.num_nodes(),), -1, dtype=torch.long)
            node_map[high_degree_nodes] = torch.arange(num_nodes_to_cache)
            node_map = node_map.to(device)

            feat_cache = g.ndata['feat'][high_degree_nodes].to(device)
            print(f"Cached features for {num_nodes_to_cache} nodes on GPU based on degree.")

    total_start_time = time.time()
    epoch_lines = train(args, device, g, dataset, model, num_classes, centrality_vals, feat_cache, node_map, cached_nodes_mask)
    total_end_time = time.time()
    
    print("\n--- Training Summary ---")
    for line in epoch_lines:
        print(line)
    
    total_execution_time = total_end_time - total_start_time
    print(f"Total script execution time: {total_execution_time:.2f} seconds")
    
    if total_execution_time < 420:
        print("\nPerformance target met: 100 epochs completed in less than 7 minutes.")
    else:
        print("\nPerformance target not met: 100 epochs took 7 minutes or more.")

    test_mask = g.ndata['test_mask']
    test_idx = torch.nonzero(test_mask).squeeze()
    acc = layerwise_infer(device, g, test_idx, model, num_classes, batch_size=4096)
    print("Test Accuracy {:.4f}".format(acc.item()))