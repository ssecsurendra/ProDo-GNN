
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
from sklearn.metrics import f1_score
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
import dgl.function as fn


def optimized_dominating_set(g, train_idx, top_train_idx):
    """
    Optimized dominating set computation using DGL message passing.
    """
    device = top_train_idx.device
    num_nodes = g.num_nodes()
    
    # A large float value for nodes not in top_train_idx
    large_float = float(len(top_train_idx))
    pos_map = torch.full((num_nodes,), large_float, dtype=torch.float32, device=device)
    pos_map[top_train_idx] = torch.arange(len(top_train_idx), device=device, dtype=torch.float32)

    # Subgraph induced by training nodes
    g_train = g.subgraph(train_idx)
    
    # The initial minimum position for each node is its own position
    g_train.ndata['min_pos'] = pos_map[train_idx]

    # Message passing to find the minimum position in the neighborhood
    g_train.update_all(fn.copy_u('min_pos', 'm'), fn.min('m', 'min_neighbor_pos'))
    
    # The final minimum position is the minimum of the node's own position and its neighbors' minimum positions
    final_min_pos = torch.min(g_train.ndata['min_pos'], g_train.ndata['min_neighbor_pos']).long()

    # The dominating set is the set of unique nodes corresponding to the final minimum positions
    dominating_nodes = top_train_idx[final_min_pos]
    dominating_set = torch.unique(dominating_nodes)
    
    return dominating_set


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
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
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
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    y_true = torch.cat(ys).cpu().numpy()
    y_pred = torch.cat(y_hats).sigmoid().cpu().numpy() > 0.5
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')        
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multilabel",
        num_labels=num_classes,
        threshold=0.5
    ),f1_micro,f1_macro


def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size)
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        y_true = label.cpu().numpy()
        y_pred = pred.sigmoid().cpu().numpy() > 0.5
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        return MF.accuracy(
            pred, label, 
            task="multilabel",
            num_labels=num_classes,
            threshold=0.5
        ),f1_micro,f1_macro

def train(args, device, g, dataset, model, num_classes, centrality_vals):
    train_mask=g.ndata['train_mask']
    val_mask=g.ndata['val_mask']
    train_idx = torch.nonzero(train_mask).squeeze().to(device)
    
    train_degrees = centrality_vals[train_idx]  
    sorted_idx = cp.argsort(-train_degrees)  
    k = int(1.0 * len(train_idx))  
    top_train_idx = train_idx[sorted_idx[:k]]
    
    dominating_start_time = time.time()
    dominating_set = optimized_dominating_set(g, train_idx, top_train_idx)
    dominating_end_time = time.time()
    
    print(f"Dominating set computation time: {dominating_end_time - dominating_start_time:.4f} seconds")
    print("Original training nodes:", len(train_idx))
    print("Number of nodes in dominating set:", len(dominating_set))

    target_train_size = int(args.target_train_percentage * len(train_idx))
    final_train_idx_set = set(dominating_set.cpu().tolist())
    
    Maintaining_training_start = time.time()
    if len(final_train_idx_set) < target_train_size:
        top_train_idx_list = top_train_idx.cpu().tolist()
        for node in top_train_idx_list:
            if len(final_train_idx_set) >= target_train_size:
                break
            final_train_idx_set.add(node)
    maintaining_training_end = time.time()
    print(f"Training node maintaining time: {maintaining_training_end - Maintaining_training_start:.4f} seconds")
    
    final_train_idx = torch.tensor(list(final_train_idx_set), device=device)
    print(f"Final training nodes (Dominating Set + Top Centrality): {len(final_train_idx)}")
    
    val_idx = torch.nonzero(val_mask).squeeze().to(device)
    
    sampler = NeighborSampler(
        [int(fanout) for fanout in args.fanout.split(",")],
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )

    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        final_train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    epoch_lines = []
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        start_time1 = time.time()
        
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        execution_time = time.time() - start_time1
        acc, micro, macro = evaluate(model, g, val_dataloader, num_classes)
        
        epoch_line = (f"Epoch {epoch:05d} | Loss {total_loss / (it + 1):.4f} | "
                      f"Accuracy {acc.item():.4f} | Micro-F1 {micro:.4f} | Macro-F1 {macro:.4f} | Time {execution_time:.4f}")
        print(epoch_line)
        epoch_lines.append(epoch_line)
        
    return epoch_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="puregpu", choices=["cpu", "mixed", "puregpu"])
    parser.add_argument("--dt", type=str, default="float", help="data type(float, bfloat16)")
    parser.add_argument("--dataset", type=str, default="yelp")
    parser.add_argument("--fanout", type=str, default="20,20,20")
    parser.add_argument("--target_train_percentage", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    if args.dataset == "yelp":
        dataset = YelpDataset()
        centrality_file = 'dissimilarity-eigenvector-centrality/yelp_dissimilar_eigen-centrality.npy'
        sortedcol_file = 'dissimilarity-eigenvector-centrality/yelp_dissimilar_eigen_sorted-col-index.npy'
    else:
        raise ValueError("This script is configured for Yelp dataset only.")

    G = dataset[0]
    test_mask = G.ndata['test_mask']
    test_idx = torch.nonzero(test_mask).squeeze()
    
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    G = G.to(device)
    
    indptr, indices, edge_ids = G.adj_tensors('csr')
    
    print("Loading binary .npy files...")
    sorted_col_idx = torch.from_numpy(np.load(sortedcol_file)).to(device)
    centrality_vals = torch.from_numpy(np.load(centrality_file)).to(device)
    print("Files loaded.")
    
    assert sorted_col_idx.shape == indices.shape
    num_nodes = len(indptr) - 1
    row_ids = torch.searchsorted(indptr[1:], torch.arange(len(indices), device=device), right=True)
    col_ids = sorted_col_idx.to(torch.int64)

    g = dgl.graph((row_ids, col_ids), num_nodes=num_nodes)
    g.ndata.update(G.ndata)
    if len(G.edata) > 0:
        g.edata.update(G.edata)
    del G
    torch.cuda.empty_cache()

    assert len(centrality_vals) == g.num_nodes(), "Mismatch between centrality file and graph nodes"

    num_classes = dataset.num_classes
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        model = model.to(dtype=torch.bfloat16)

    print("Training...")
    epoch_lines = train(args, device, g, dataset, model, num_classes, centrality_vals)

    print("Testing...")
    acc, f1_micro, f1_macro = layerwise_infer(device, g, test_idx, model, num_classes, batch_size=4096)
    
    test_results = (f"\nTest Accuracy: {acc.item():.4f}\n"
                    f"Test Micro-F1: {f1_micro:.4f}\n"
                    f"Test Macro-F1: {f1_macro:.4f}")
    print(test_results)
    epoch_lines.append(test_results)
    
    with open('epoch_data_optimized_yelp.txt', 'w') as file:
        for value in epoch_lines:
            file.write(str(value) + '\n')
