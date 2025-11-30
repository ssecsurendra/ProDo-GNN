import dgl
import argparse
import time
import matplotlib.pyplot as plt
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset,RedditDataset,FlickrDataset,YelpDataset
import numpy as np
import cupy as cp


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
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
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size)
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(pred, label, task="multiclass", num_classes=num_classes)


def train(args, device, g, dataset, model, num_classes, degree_vals):
    train_mask=g.ndata['train_mask']
    val_mask=g.ndata['val_mask']
    train_idx = torch.nonzero(train_mask).squeeze().to(device)
    
    train_degrees = degree_vals[train_idx]  
    sorted_idx = cp.argsort(-train_degrees)  
    k = int(0.7 * len(train_idx))  
    top_train_idx = train_idx[sorted_idx[:k]]  

    print("Original training nodes:", len(train_idx))
    print("Filtered training nodes (top 70%):", len(top_train_idx))
    val_idx = torch.nonzero(val_mask).squeeze().to(device)
    
    sampler = NeighborSampler(
        [int(fanout) for fanout in args.fanout.split(",")],
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g, top_train_idx, sampler, device=device,
        batch_size=args.batch_size, shuffle=True,
        drop_last=False, num_workers=0, use_uva=use_uva,
    )
    val_dataloader = DataLoader(
        g, val_idx, sampler, device=device,
        batch_size=args.batch_size, shuffle=True,
        drop_last=False, num_workers=0, use_uva=use_uva,
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
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        execution_time = time.time() - start_time1
        acc = evaluate(model, g, val_dataloader, num_classes)
        epoch_line = "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {:.4f}".format(
            epoch, total_loss / (it + 1), acc.item(), execution_time
        )
        epoch_lines.append(epoch_line)
    return epoch_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="puregpu", choices=["cpu", "mixed", "puregpu"])
    parser.add_argument("--dt", type=str, default="float")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--fanout", type=str, default="20,20,20")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=100)
    args = parser.parse_args()

    # Define file paths for binary .npy files
    if args.dataset == "ogbn-products":
        degree_file = 'eigenvector-centrality-binary/ogbn-products_eigen-centrality.npy'
        sortedcol_file = 'eigenvector-centrality-binary/ogbn-products_eigen_sorted-col-index.npy'
    elif args.dataset == "ogbn-arxiv":
        degree_file = 'eigenvector-centrality-binary/ogbn-arxiv_eigen-centrality.npy'
        sortedcol_file = 'eigenvector-centrality-binary/ogbn-arxiv_eigen_sorted-col-index.npy'
    elif args.dataset == "reddit":
        degree_file = 'eigenvector-centrality-binary/reddit_eigen-centrality.npy'
        sortedcol_file = 'eigenvector-centrality-binary/reddit_eigen_sorted-col-index.npy'
    # Add other datasets as needed...
    else:
        raise ValueError(f"Binary file paths not defined for dataset: {args.dataset}")

    # load and preprocess dataset
    if args.dataset == "ogbn-products":
        dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    # Add other dataset loaders here...
    else:
        # Fallback for other datasets if not fully configured
        print(f"Loading dataset {args.dataset}...")
        if args.dataset == "cora": dataset = CoraGraphDataset()
        elif args.dataset == "citeseer": dataset = CiteseerGraphDataset()
        elif args.dataset == "pubmed": dataset = PubmedGraphDataset()
        elif args.dataset == "reddit": dataset = RedditDataset()
        else: raise ValueError("Unknown dataset: {}".format(args.dataset))

    G = dataset[0]
    test_mask = G.ndata['test_mask']
    test_idx = torch.nonzero(test_mask).squeeze()
    
    # print("Making graph bidirected to match centrality calculation...")
    # G = dgl.to_bidirected(G, copy_ndata=True)
    
    device = torch.device("cuda" if args.mode == "puregpu" else "cpu")
    G = G.to(device)
    
    indptr, indices, edge_ids = G.adj_tensors('csr')

    # --- Load binary files using np.load ---
    print("Loading binary .npy files...")
    sorted_col_idx = torch.from_numpy(np.load(sortedcol_file)).to(device)
    degree_vals = torch.from_numpy(np.load(degree_file)).to(device)
    print("Files loaded.")

    assert sorted_col_idx.shape == indices.shape, "Shape mismatch between sorted and original indices!"
    
    num_nodes = len(indptr) - 1
    row_ids = torch.searchsorted(indptr[1:], torch.arange(indices.numel(), device=device), right=True)
    col_ids = sorted_col_idx.to(torch.int64)

    g = dgl.graph((row_ids, col_ids), num_nodes=num_nodes)
    if len(G.edata) > 0:
        g.edata['__orig__'] = torch.arange(indices.numel(), device=device)
    
    g.ndata.update(G.ndata)
    if len(G.edata) > 0:
        orig_eids = g.edata['__orig__']
        new_edata = {}
        for key, feat in G.edata.items():
            if key == '__orig__': continue
            new_edata[key] = feat[orig_eids]
        g.edata.update(new_edata)
    del G
    torch.cuda.empty_cache()

    assert len(degree_vals) == g.num_nodes(), "Mismatch between degree centrality file and graph nodes"
    
    if not torch.cuda.is_available():
        args.mode = "cpu"

    labels = g.ndata["label"]
    num_classes = int(labels.max().item()) + 1
    in_size = g.ndata["feat"].shape[1]
    out_size = num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        model = model.to(dtype=torch.bfloat16)

    epoch_lines = train(args, device, g, dataset, model, num_classes, degree_vals)

    acc = layerwise_infer(device, g, test_idx, model, num_classes, batch_size=4096)
    Accuracy = "Test Accuracy {:.4f}".format(acc.item())
    epoch_lines.append(Accuracy)
    
    with open('epoch_data_binary.txt', 'w') as file:
        for value in epoch_lines:
            file.write(str(value) + '\n')
