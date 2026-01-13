
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
import os

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
        return MF.accuracy(
            pred, label, task="multiclass", num_classes=num_classes
        )


def train(args, device, g, dataset, model, num_classes, centrality_vals, dominating_set_nodes):
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    train_idx = torch.nonzero(train_mask).squeeze().to(device)
    
    # Create a boolean mask from the loaded dominating set for efficient filtering
    dom_set_mask = torch.zeros(g.num_nodes(), dtype=torch.bool, device=device)
    dom_set_mask[dominating_set_nodes] = True

    # The final training nodes are the intersection of the original train_idx
    # and the nodes in the dominating set.
    final_train_idx = train_idx[dom_set_mask[train_idx]]
    
    print(f"Original training nodes: {len(train_idx)}")
    print(f"Nodes in loaded dominating set: {len(dominating_set_nodes)}")
    print(f"Final training nodes (intersection): {len(final_train_idx)}")

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
    
    total_training_time = 0.0
    total_for_loop_time = 0.0
    total_model_time = 0.0
    epoch_lines = []
    total_src_nodes_layer_3=0
    total_src_nodes_layer_2=0
    total_src_nodes_layer_1=0

    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        execution_time = 0.0
        iteration_time1 =0.0
        model_time1 =0.0
        x_y_time = 0.0
        pred_time = 0.0
        loss_time = 0.0
        backward_time = 0.0
        optim_time = 0.0
        start_time1 = time.time()
        iter_time = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            iteration_time = time.time() - iter_time
            batch_start_time = time.time()

            start_x_y_time = time.time()
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            if epoch == 0:
                total_src_nodes_layer_3 = total_src_nodes_layer_3 + blocks[0].num_src_nodes()
                total_src_nodes_layer_2 = total_src_nodes_layer_2 + blocks[1].num_src_nodes()
                total_src_nodes_layer_1 = total_src_nodes_layer_1 + blocks[-1].num_src_nodes()
            end_x_y_time = time.time()

            start_pred_time = time.time()
            y_hat = model(blocks, x)
            end_pred_time = time.time()

            start_loss_time = time.time()
            loss = F.cross_entropy(y_hat, y)
            end_loss_time = time.time()

            start_backward_time = time.time()
            opt.zero_grad()
            loss.backward()
            end_backward_time = time.time()

            start_optim_time = time.time()
            opt.step()
            end_optim_time = time.time()

            total_loss += loss.item()
            batch_end_time = time.time()
            model_time = batch_end_time - batch_start_time
            iteration_time1 += iteration_time
            model_time1 += model_time
            x_y_time1 = end_x_y_time - start_x_y_time
            x_y_time += x_y_time1
            pred_time1 = end_pred_time - start_pred_time
            pred_time += pred_time1
            backward_time1 = end_backward_time - start_backward_time
            backward_time += backward_time1
            optim_time1 = end_optim_time - start_optim_time
            optim_time += optim_time1
            iter_time = time.time()

        end_time1 = time.time()
        execution_time = end_time1 - start_time1
        total_training_time += execution_time
        total_for_loop_time += iteration_time1
        total_model_time += model_time1
        if epoch == 0:
            layer_line = "Layer_1 {:d} | Layer_2 {:d} | Layer_3 {:d}" .format(int(total_src_nodes_layer_1/(it+1)), int(total_src_nodes_layer_2/(it+1)), int(total_src_nodes_layer_3/(it+1)))
            epoch_lines.append(layer_line)
        acc = evaluate(model, g, val_dataloader, num_classes)
        
        epoch_line = "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {:.4f} | Loop_Time : {:.4f} | Model_Time : {:.4f} | x_y_time : {:.4f} | pred_time : {:.4f} | backward_time : {:.4f} | optim_time : {:.4f} ".format(epoch, total_loss / (it + 1), acc.item(), execution_time, iteration_time1, model_time1, x_y_time, pred_time, backward_time, optim_time )
        print(epoch_line)
        epoch_lines.append(epoch_line)

    tt_time = "Sampling time: {:.4f}, Model training time: {:.4f}, Total time {:.4f}".format(total_for_loop_time, total_model_time, total_training_time)
    epoch_lines.append(tt_time)     
    return epoch_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="puregpu", choices=["cpu", "mixed", "puregpu"])
    parser.add_argument("--dt", type=str, default="float", help="data type(float, bfloat16)")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv",
                        choices=['reddit', 'ogbn-products', 'ogbn-arxiv', 'igb-small'])
    parser.add_argument("--fanout", type=str, default="20,20,20")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode for {args.dataset} dataset.")

    # Define file paths based on dataset
    base_dir = "dissimilarity-eigenvector-centrality"
    centrality_file = os.path.join(base_dir, f"{args.dataset}_dissimilar_eigen-centrality.npy")
    sortedcol_file = os.path.join(base_dir, f"{args.dataset}_dissimilar_eigen_sorted-col-index.npy")
    dom_set_file = os.path.join(base_dir, f"{args.dataset}_eigen_dominating_set.npy")

    # Load dataset
    if args.dataset == "reddit":
        dataset = RedditDataset()
    elif args.dataset == "ogbn-products":
        dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    elif args.dataset == "ogbn-arxiv":
        dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv"))
    elif args.dataset == "igb-small":
        load_path = './dataset/igb_small.dgl'
        dataset, _ = dgl.load_graphs(load_path)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    G = dataset[0]
    
    if args.dataset.startswith('ogbn') or args.dataset.startswith('igb'):
        # OGBN and IGB datasets have their own splits
        train_idx_split = dataset.train_idx if hasattr(dataset, 'train_idx') else G.ndata['train_mask'].nonzero().squeeze()
        val_idx_split = dataset.val_idx if hasattr(dataset, 'val_idx') else G.ndata['val_mask'].nonzero().squeeze()
        test_idx_split = dataset.test_idx if hasattr(dataset, 'test_idx') else G.ndata['test_mask'].nonzero().squeeze()
        G.ndata['train_mask'] = torch.zeros(G.num_nodes(), dtype=torch.bool)
        G.ndata['val_mask'] = torch.zeros(G.num_nodes(), dtype=torch.bool)
        G.ndata['test_mask'] = torch.zeros(G.num_nodes(), dtype=torch.bool)
        G.ndata['train_mask'][train_idx_split] = True
        G.ndata['val_mask'][val_idx_split] = True
        G.ndata['test_mask'][test_idx_split] = True
    
    test_mask = G.ndata['test_mask']
    test_idx = torch.nonzero(test_mask).squeeze()
    
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    G = G.to(device)
    
    indptr, indices, edge_ids = G.adj_tensors('csr')
    
    print("Loading pre-computed files...")
    centrality_vals = torch.from_numpy(np.load(centrality_file)).to(device)
    sorted_col_idx = torch.from_numpy(np.load(sortedcol_file)).to(device)
    dominating_set_nodes = torch.from_numpy(np.load(dom_set_file)).to(device)
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

    labels = g.ndata["label"]
    num_classes = int(labels.max().item()) + 1
    in_size = g.ndata["feat"].shape[1]
    out_size = num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        model = model.to(dtype=torch.bfloat16)

    epoch_lines = train(args, device, g, dataset, model, num_classes, centrality_vals, dominating_set_nodes)

    print("\n--- Final Testing ---")
    acc = layerwise_infer(device, g, test_idx, model, num_classes, batch_size=4096)
    
    test_results = f"\nTest Accuracy: {acc.item():.4f}"
    print(test_results)
    epoch_lines.append(test_results)
    
    output_filename = f'epoch_data_trained_on_dom_set_{args.dataset}.txt'
    with open(output_filename, 'w') as file:
        for value in epoch_lines:
            file.write(str(value) + '\n')
    print(f"\nResults saved to {output_filename}")
